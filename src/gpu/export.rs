//! Explicit image exports use an asynchronous GPU readback and PNG worker.
//! Neither GPU completion nor compression waits on the application event thread.
use anyhow::{Context, Result, bail};
use std::{
    path::Path,
    sync::mpsc,
    time::{Duration, Instant},
};

pub struct ImageExport {
    result: mpsc::Receiver<Result<(), String>>,
}
impl ImageExport {
    pub fn poll(&self) -> Option<Result<()>> {
        match self.result.try_recv() {
            Ok(result) => Some(result.map_err(anyhow::Error::msg)),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::anyhow!("Export worker stopped")))
            }
        }
    }
}

pub(super) fn start(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    path: &Path,
) -> Result<ImageExport> {
    let width = texture.width();
    let height = texture.height();
    let row = (width * 4).div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Asynchronous PNG readback"),
        size: u64::from(row) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (mapped, receive) = mpsc::sync_channel(1);
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = mapped.send(r);
    });
    let (send, result) = mpsc::sync_channel(1);
    let device = device.clone();
    let path = path.to_owned();
    std::thread::Builder::new()
        .name("PNG export".into())
        .spawn(move || {
            let result = (|| -> Result<()> {
                let started = Instant::now();
                loop {
                    match receive.recv_timeout(Duration::from_millis(5)) {
                        Ok(result) => {
                            result?;
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            device.poll(wgpu::PollType::Poll)?;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            bail!("PNG readback callback stopped")
                        }
                    }
                    if started.elapsed() > Duration::from_secs(120) {
                        bail!("PNG readback timed out");
                    }
                }
                let view = staging.slice(..).get_mapped_range()?;
                let data: Vec<u8> = view
                    .chunks_exact(row as usize)
                    .flat_map(|r| r[..width as usize * 4].iter().copied())
                    .collect();
                drop(view);
                staging.unmap();
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)?;
                }
                let writer = std::io::BufWriter::new(std::fs::File::create(&path)?);
                let mut png = png::Encoder::new(writer, width, height);
                png.set_color(png::ColorType::Rgba);
                png.set_depth(png::BitDepth::Eight);
                png.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
                png.write_header()?.write_image_data(&data)?;
                Ok(())
            })();
            let _ = send.send(result.map_err(|e| format!("{e:#}")));
        })
        .context("Start PNG export worker")?;
    Ok(ImageExport { result })
}
