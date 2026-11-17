//! Metal 4 TensorOps for the two cache MLP shapes. The native shader runs inside
//! wgpu's command stream, between WGSL feature packing and result application.
use anyhow::{Result, ensure};

pub(super) struct Inference {
    pub layout: wgpu::BindGroupLayout,
    pub group: wgpu::BindGroup,
    #[cfg(test)]
    pub data: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}
impl Inference {
    pub fn new(device: &wgpu::Device, pixels: u32, hidden: u32) -> Result<Self> {
        ensure!(
            device
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS),
            "Native Metal shaders unavailable"
        );
        ensure!(
            hidden == 32 || hidden == 64,
            "Unsupported tensor model width"
        );
        let capacity = (pixels * 2).div_ceil(32) * 32;
        let parameters = 2 * hidden * hidden + 5 * hidden + 3;
        let size = u64::from(8 + parameters + capacity * (hidden + 4)) * 4;
        ensure!(
            size <= device.limits().max_storage_buffer_binding_size,
            "Tensor query storage exceeds device limits"
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Metal TensorOps inference"),
            entries: &[super::buffer_entry(
                0,
                wgpu::BufferBindingType::Storage { read_only: false },
            )],
        });
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor weights, encoded queries and predictions"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Metal TensorOps inference"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: data.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Native tensor inference"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source =
            include_str!("../shaders/tensor.metal").replace("@WIDTH@", &hidden.to_string());
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        // This source is owned by the renderer. Its sole storage binding and
        // 128-thread scope exactly match the explicit layout and TensorOp scope.
        let shader = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("Metal 4 neural TensorOps"),
                msl: Some(source.into()),
                entry_points: vec![wgpu::PassthroughShaderEntryPoint {
                    name: "tensor_inference".into(),
                    workgroup_size: (128, 1, 1),
                }]
                .into(),
                ..Default::default()
            })
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Metal 4 neural TensorOps"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tensor_inference"),
            compilation_options: Default::default(),
            cache: None,
        });
        let validation = pollster::block_on(errors.pop());
        let internal = pollster::block_on(internal.pop());
        if let Some(error) = validation.or(internal) {
            anyhow::bail!("Metal TensorOps unavailable: {error}");
        }
        Ok(Self {
            layout,
            group,
            #[cfg(test)]
            data,
            pipeline,
        })
    }
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let groups = count.div_ceil(32);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
    }
}

pub(super) struct Cache {
    pub inference: Inference,
    pack: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    parameters: u32,
}
impl Cache {
    pub fn new(
        device: &wgpu::Device,
        base: &wgpu::BindGroupLayout,
        queries: &wgpu::BindGroupLayout,
        pixels: u32,
        hidden: u32,
        source: String,
    ) -> Option<Self> {
        if !super::metal::tensor_accelerators(device) {
            return None;
        }
        let inference = match Inference::new(device, pixels, hidden) {
            Ok(i) => i,
            Err(error) => {
                eprintln!("Tensor inference fallback: {error:#}");
                return None;
            }
        };
        let source = format!(
            "{source}\n{}\n{}",
            include_str!("../shaders/tensor.wgsl"),
            if hidden == 32 {
                include_str!("../shaders/nrc_tensor.wgsl")
            } else {
                include_str!("../shaders/photon_nrc_tensor.wgsl")
            }
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tensor feature packing and prediction application"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tensor cache integration"),
            bind_group_layouts: &[Some(base), Some(queries), Some(&inference.layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Some(Self {
            pack: pipeline("tensor_pack"),
            apply: pipeline("tensor_apply"),
            inference,
            parameters: 2 * hidden * hidden + 5 * hidden + 3,
        })
    }
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        queries: &wgpu::BindGroup,
        count: u32,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pack);
            pass.set_bind_group(0, base, &[]);
            pass.set_bind_group(1, queries, &[]);
            pass.set_bind_group(2, &self.inference.group, &[]);
            pass.dispatch_workgroups(count.max(self.parameters).div_ceil(64), 1, 1);
        }
        self.inference.dispatch(encoder, count);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.apply);
        pass.set_bind_group(0, base, &[]);
        pass.set_bind_group(1, queries, &[]);
        pass.set_bind_group(2, &self.inference.group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Metal 4 TensorOps GPU"]
    fn native_tensor_mlp_matches_float_reference_for_both_model_shapes() {
        let (device, queue) = pollster::block_on(crate::gpu::headless_device()).unwrap();
        let r = crate::gpu::Renderer::new(device, queue, 8, 8, 42).unwrap();
        for h in [32, 64] {
            let t = Inference::new(&r.device, 64, h).unwrap();
            let count = 47u32;
            let capacity = 128u32;
            let parameters = 2 * h * h + 5 * h + 3;
            let masks = 8 + parameters;
            let features = masks + capacity;
            let predictions = features + capacity * h;
            let mut data = vec![0_f32; t.data.size() as usize / 4];
            for (i, v) in [
                count,
                h,
                8,
                masks,
                features,
                predictions,
                1000_f32.to_bits(),
                0,
            ]
            .into_iter()
            .enumerate()
            {
                data[i] = f32::from_bits(v);
            }
            for i in 0..parameters {
                data[(8 + i) as usize] = ((i * 719 % 101) as f32 / 100. - 0.5) * 0.1;
            }
            for q in 0..count {
                data[(masks + q) as usize] = 1.;
                for j in 0..h {
                    data[(features + q * h + j) as usize] = ((q * 13 + j * 23) % 101) as f32 / 100.;
                }
            }
            r.queue
                .write_buffer(&t.data, 0, bytemuck::cast_slice(&data));
            let mut encoder = r.device.create_command_encoder(&Default::default());
            t.dispatch(&mut encoder, count);
            let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(count * 3 * 4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                &t.data,
                u64::from(predictions) * 4,
                &stage,
                0,
                stage.size(),
            );
            let actual = r.read_buffer(encoder, &stage).unwrap();
            let actual: Vec<f32> = actual
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let mut max_error = 0_f32;
            for q in 0..count {
                let mut a =
                    data[(features + q * h) as usize..(features + (q + 1) * h) as usize].to_vec();
                for layer in 0..2 {
                    let w = 8 + layer * (h * h + h);
                    a = (0..h)
                        .map(|i| {
                            (data[(w + h * h + i) as usize]
                                + (0..h)
                                    .map(|j| a[j as usize] * data[(w + j * h + i) as usize])
                                    .sum::<f32>())
                            .max(0.)
                        })
                        .collect();
                }
                let w = 8 + 2 * h * h + 2 * h;
                for c in 0..3 {
                    let expected = (data[(w + 3 * h + c) as usize]
                        + (0..h)
                            .map(|j| a[j as usize] * data[(w + j * 3 + c) as usize])
                            .sum::<f32>())
                    .clamp(0., 1000.);
                    max_error = max_error.max((expected - actual[(q * 3 + c) as usize]).abs());
                }
            }
            eprintln!("Tensor MLP width {h}: maximum RGB error {max_error}");
            assert!(
                max_error < 0.001,
                "Native tensor model disagrees with float reference"
            );
        }
    }
}
