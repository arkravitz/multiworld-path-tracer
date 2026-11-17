//! GPU-resident cache: frozen rendering weights, parallel minibatch gradients,
//! ordered Adam reduction, and bias-corrected EMA publication.
pub(super) const PARAMETERS: u32 = 2211;
/// 16-byte state + three parameter arrays, padded to a sample's vec4 alignment.
pub(super) const SAMPLE_OFFSET: u64 = (16 + 3 * PARAMETERS as u64 * 4).div_ceil(16) * 16;
pub(super) const BATCH: u32 = 128;
pub(super) const TRAINING_OFFSET: u64 = SAMPLE_OFFSET + 1024 * 64;
pub(super) const BYTES: u64 =
    (TRAINING_OFFSET + PARAMETERS as u64 * (BATCH as u64 + 1) * 4 + BATCH as u64 * 4).div_ceil(16)
        * 16
        + 16
        + 256 * 64
        + 1024 * 16;
pub(super) struct Neural {
    pub tensor: Option<super::tensor::Cache>,
    pub initialized: bool,
    pub batches: u32,

    pub queries: wgpu::BindGroup,
    resolve: Option<[wgpu::ComputePipeline; 6]>,
    finish_guides: Option<[wgpu::ComputePipeline; 6]>,
    resolve_teachers: Option<wgpu::ComputePipeline>,
    init: wgpu::ComputePipeline,
    teach: [wgpu::ComputePipeline; 6],
    train: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    trace: [wgpu::ComputePipeline; 6],
    trace_guides: [wgpu::ComputePipeline; 6],
}
impl Neural {
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        base: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        trace_group_y: u32,
    ) -> Self {
        let cooperative = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let query_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Neural ray queries"),
            entries: &[super::buffer_entry(
                3,
                wgpu::BufferBindingType::Storage { read_only: false },
            )],
        });
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neural query queue"),
            size: if cooperative {
                u64::from(width) * u64::from(height) * 128
            } else {
                64
            },
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let queries = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Neural query queue"),
            layout: &query_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 3,
                resource: query_buffer.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Neural kernels"),
            bind_group_layouts: &[Some(base), Some(&query_layout)],
            immediate_size: 0,
        });
        let matrix_source = super::ray_query::shader(
            device,
            format!(
                "enable wgpu_cooperative_matrix;\n{}\n{}\n{}\n{}\n{}",
                concat!(
                    concat!(
                        include_str!("../shaders/trace.wgsl"),
                        "\n",
                        include_str!("../shaders/adaptive.wgsl")
                    ),
                    "\n",
                    include_str!("../shaders/caustics_stub.wgsl")
                ),
                include_str!("../shaders/nrc.wgsl"),
                include_str!("../shaders/nrc_matrix.wgsl"),
                include_str!("../shaders/nrc_queued.wgsl"),
                include_str!("../shaders/denoise.wgsl")
            ),
        );
        let matrix_shader = cooperative.then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Metal cooperative neural matrices"),
                source: wgpu::ShaderSource::Wgsl(matrix_source.clone().into()),
            })
        });
        let tensor = if cooperative {
            super::tensor::Cache::new(
                device,
                base,
                &query_layout,
                width * height,
                32,
                matrix_source,
            )
        } else {
            None
        };
        let pipeline = |entry: &'static str, mode: u32, guides: bool| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: if entry == "neural_resolve_teachers"
                    || entry == "neural_resolve_queries"
                    || entry == "neural_finish_guides"
                    || (entry == "accumulate" && cooperative)
                {
                    matrix_shader.as_ref().unwrap_or(shader)
                } else {
                    shader
                },
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("NRC_ENABLED", 1.),
                        (
                            "NRC_QUEUED",
                            u32::from(cooperative && entry == "accumulate") as f64,
                        ),
                        (
                            "NRC_TEACHER",
                            u32::from(entry == "neural_teach" || entry == "neural_resolve_teachers")
                                as f64,
                        ),
                        (
                            "NRC_TEACHER_QUEUED",
                            u32::from(cooperative && entry == "neural_teach") as f64,
                        ),
                        ("NRC_COOPERATIVE", 0.),
                        ("TRACE_GROUP_Y", trace_group_y as f64),
                        ("ACTIVE_MODE", mode as f64),
                        ("DENOISE_INPUT", u32::from(guides) as f64),
                    ],
                    // Every shared activation/weight lane is written before a
                    // barrier and read. Avoid serial backend zero-filling.
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        Self {
            tensor,
            resolve: cooperative.then(|| {
                std::array::from_fn(|m| pipeline("neural_resolve_queries", m as u32, false))
            }),
            finish_guides: cooperative.then(|| {
                std::array::from_fn(|m| pipeline("neural_finish_guides", m as u32, false))
            }),
            resolve_teachers: cooperative.then(|| pipeline("neural_resolve_teachers", 0, false)),
            queries,
            initialized: false,
            batches: 0,

            init: pipeline("neural_initialize", 0, false),
            teach: std::array::from_fn(|m| pipeline("neural_teach", m as u32, false)),
            train: pipeline("neural_train_fused", 0, false),
            apply: pipeline("neural_apply_gradients", 0, false),
            finish: pipeline("neural_finish_training", 0, false),
            trace: std::array::from_fn(|m| pipeline("accumulate", m as u32, false)),
            trace_guides: std::array::from_fn(|m| pipeline("accumulate", m as u32, true)),
        }
    }
    #[cfg(test)]
    pub fn resolve_pipeline(&self, mode: usize) -> &wgpu::ComputePipeline {
        &self.resolve.as_ref().unwrap()[mode]
    }
    pub fn resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        mode: usize,
        size: [u32; 2],
        guides: bool,
        use_tensor: bool,
    ) {
        let [width, height] = size;
        if use_tensor && let Some(tensor) = &self.tensor {
            tensor.run(
                encoder,
                group,
                &self.queries,
                width * height * if mode == 2 { 2 } else { 1 },
            );
            if guides {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.finish_guides.as_ref().unwrap()[mode]);
                pass.set_bind_group(0, group, &[]);
                pass.set_bind_group(1, &self.queries, &[]);
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }
            return;
        }
        if let Some(pipelines) = &self.resolve {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines[mode]);
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(1, &self.queries, &[]);
            let groups = (width * height * if mode == 2 { 2 } else { 1 }).div_ceil(32);
            pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
            if guides {
                pass.set_pipeline(&self.finish_guides.as_ref().unwrap()[mode]);
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }
        }
    }
    pub fn reset(&mut self) {
        self.initialized = false;
        self.batches = 0;
    }
    pub fn initialize(&mut self, encoder: &mut wgpu::CommandEncoder, group: &wgpu::BindGroup) {
        if self.initialized {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, &self.queries, &[]);
        pass.set_pipeline(&self.init);
        pass.dispatch_workgroups(PARAMETERS.div_ceil(64), 1, 1);
        self.initialized = true;
    }
    pub fn trace_pipeline(&self, mode: usize, guides: bool) -> &wgpu::ComputePipeline {
        if guides {
            &self.trace_guides[mode]
        } else {
            &self.trace[mode]
        }
    }
    pub fn learn(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        mode: usize,
        self_training: bool,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Anchored self-training and parallel neural minibatches"),
            ..Default::default()
        });
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, &self.queries, &[]);
        pass.set_pipeline(&self.teach[mode]);
        pass.dispatch_workgroups(4, 1, 1);
        if self_training && let Some(resolve) = &self.resolve_teachers {
            pass.set_pipeline(resolve);
            pass.dispatch_workgroups(8, 1, 1);
        }
        for _ in 0..4 {
            pass.set_pipeline(&self.train);
            pass.dispatch_workgroups(BATCH, 1, 1);
            pass.set_pipeline(&self.apply);
            pass.dispatch_workgroups(PARAMETERS.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.finish);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.batches = self.batches.wrapping_add(1);
    }
}
