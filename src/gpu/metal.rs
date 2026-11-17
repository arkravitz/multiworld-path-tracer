//! Owned Metal/wgpu sharing boundary. Foreign work uses a separate command
//! queue, starts only after wgpu completion, and completes before publication.
#[cfg(target_os = "macos")]
use anyhow::Context;
use anyhow::Result;
#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;

pub(super) fn ray_query_hardware(adapter: &wgpu::Adapter) -> bool {
    if adapter.get_info().backend != wgpu::Backend::Metal {
        return true;
    }
    #[cfg(target_os = "macos")]
    {
        let Some(hal) = (unsafe { adapter.as_hal::<wgpu::hal::api::Metal>() }) else {
            return false;
        };
        // Apple9 is the first Mac GPU family with dedicated ray tracing (M3).
        hal.raw_device()
            .supportsFamily(objc2_metal::MTLGPUFamily::Apple9)
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

pub(super) fn tensor_accelerators(device: &wgpu::Device) -> bool {
    #[cfg(target_os = "macos")]
    {
        let Some(hal) = (unsafe { device.as_hal::<wgpu::hal::api::Metal>() }) else {
            return false;
        };
        hal.raw_device()
            .supportsFamily(objc2_metal::MTLGPUFamily::Apple10)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = device;
        false
    }
}

#[derive(Clone)]
pub(super) struct SharedBuffer {
    pub buffer: wgpu::Buffer,
    #[cfg(target_os = "macos")]
    native: Retained<ProtocolObject<dyn MTLBuffer>>,
    #[cfg(target_os = "macos")]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}
// Metal buffers and command queues are thread safe. Queue ownership and buffer
// access are additionally serialized by the bounded reconstruction job channel.
#[cfg(target_os = "macos")]
unsafe impl Send for SharedBuffer {}
#[cfg(target_os = "macos")]
unsafe impl Sync for SharedBuffer {}
impl SharedBuffer {
    pub fn new(device: &wgpu::Device, size: u64) -> Result<Self> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (device, size);
            anyhow::bail!("Metal sharing requires macOS")
        }
        #[cfg(target_os = "macos")]
        {
            // The HAL guard is held only while creating resources from this exact
            // device; no foreign work is submitted through wgpu's command queue.
            let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
                .context("Not a Metal device")?;
            let native = hal
                .raw_device()
                .newBufferWithLength_options(size as usize, MTLResourceOptions::StorageModeShared)
                .context("Metal shared buffer allocation failed")?;
            let queue = hal
                .raw_device()
                .newCommandQueue()
                .context("Metal reconstruction queue allocation failed")?;
            // create_buffer_from_hal requires initialized storage. Neither API can
            // access this allocation until construction finishes.
            unsafe {
                native
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .write_bytes(0, size as usize)
            };
            let raw = unsafe { wgpu::hal::metal::Device::buffer_from_raw(native.clone(), size) };
            drop(hal);
            let buffer = unsafe {
                device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                    raw,
                    &wgpu::BufferDescriptor {
                        label: Some("Shared Metal reconstruction storage"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
            };
            Ok(Self {
                buffer,
                native,
                queue,
            })
        }
    }
    pub fn native_buffer(&self) -> *mut c_void {
        #[cfg(target_os = "macos")]
        {
            Retained::as_ptr(&self.native).cast_mut().cast()
        }
        #[cfg(not(target_os = "macos"))]
        {
            std::ptr::null_mut()
        }
    }
    pub fn native_queue(&self) -> *mut c_void {
        #[cfg(target_os = "macos")]
        {
            Retained::as_ptr(&self.queue).cast_mut().cast()
        }
        #[cfg(not(target_os = "macos"))]
        {
            std::ptr::null_mut()
        }
    }
}
