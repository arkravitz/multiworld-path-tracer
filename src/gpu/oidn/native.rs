//! Small ownership wrapper around the published OIDN 2 C ABI. Library symbols
//! remain loaded until all filters, buffers and the device have been released.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    path::PathBuf,
};
pub(super) struct Reconstruction {
    pub color: Vec<[f32; 3]>,
    pub halves: Option<[Vec<[f32; 3]>; 2]>,
}
type Handle = *mut c_void;
macro_rules! api {
    ($($name:ident: ($($arg:ty),*) -> $ret:ty),* $(,)?) => {
        struct Api { $($name: unsafe extern "C" fn($($arg),*) -> $ret,)* _library: Library }
        impl Api {
            unsafe fn load(library: Library) -> Result<Self> {
                // The exact signatures below are from OIDN 2's public oidn.h.
                Ok(Self { $($name: unsafe { *library.get(concat!(stringify!($name), "\0").as_bytes())? },)* _library: library })
            }
        }
    }
}
#[allow(non_snake_case)]
mod symbols {
    use super::*;
    api! {
        oidnNewDevice: (i32) -> Handle,
        oidnCommitDevice: (Handle) -> (),
        oidnGetDeviceError: (Handle, *mut *const c_char) -> i32,
        oidnReleaseDevice: (Handle) -> (),
        oidnNewBuffer: (Handle, usize) -> Handle,
        oidnReleaseBuffer: (Handle) -> (),
        oidnWriteBuffer: (Handle, usize, usize, *const c_void) -> (),
        oidnReadBuffer: (Handle, usize, usize, *mut c_void) -> (),
        oidnNewFilter: (Handle, *const c_char) -> Handle,
        oidnReleaseFilter: (Handle) -> (),
        oidnSetFilterImage: (Handle, *const c_char, Handle, i32, usize, usize, usize, usize, usize) -> (),
        oidnSetFilterBool: (Handle, *const c_char, bool) -> (),
        oidnSetFilterInt: (Handle, *const c_char, i32) -> (),
        oidnCommitFilter: (Handle) -> (),
        oidnExecuteFilter: (Handle) -> (),
    }
    pub struct Engine {
        api: Api,
        device: Handle,
        buffers: Vec<Handle>,
        filters: Vec<Handle>,
        size: [u32; 2],
    }
    impl Engine {
        pub fn new() -> Result<Self> {
            let mut paths = Vec::new();
            if let Some(path) = std::env::var_os("OIDN_LIBRARY_PATH") {
                paths.push(PathBuf::from(path));
            }
            if let Ok(exe) = std::env::current_exe()
                && let Some(dir) = exe.parent()
            {
                paths.push(dir.join("../Frameworks/libOpenImageDenoise.2.dylib"));
            }
            paths.push(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join(".deps/oidn-2.5.1.arm64.macos/lib/libOpenImageDenoise.2.dylib"),
            );
            paths.extend(
                [
                    "libOpenImageDenoise.2.dylib",
                    "libOpenImageDenoise.so.2",
                    "OpenImageDenoise.dll",
                ]
                .map(PathBuf::from),
            );
            let mut failures = Vec::new();
            for path in paths {
                // A user-selected or bundled library is executable native code.
                let library = match unsafe { Library::new(&path) } {
                    Ok(lib) => lib,
                    Err(e) => {
                        failures.push(format!("{}: {e}", path.display()));
                        continue;
                    }
                };
                let api = unsafe { Api::load(library) }
                    .context("OIDN 2 runtime has an incompatible ABI")?;
                // Select Metal explicitly on macOS: do not silently claim GPU
                // denoising after falling back to the CPU.
                let device =
                    unsafe { (api.oidnNewDevice)(if cfg!(target_os = "macos") { 5 } else { 0 }) };
                let engine = Self {
                    api,
                    device,
                    buffers: Vec::new(),
                    filters: Vec::new(),
                    size: [0; 2],
                };
                engine.check()?;
                ensure!(!device.is_null(), "OIDN did not create a device");
                unsafe {
                    (engine.api.oidnCommitDevice)(device);
                }
                engine.check()?;
                return Ok(engine);
            }
            bail!(
                "OIDN runtime unavailable. Run scripts/install-oidn.sh on Apple Silicon, or set OIDN_LIBRARY_PATH to an OIDN 2 library. {}",
                failures.join("; ")
            )
        }
        fn check(&self) -> Result<()> {
            let mut message = std::ptr::null();
            let code = unsafe { (self.api.oidnGetDeviceError)(self.device, &mut message) };
            if code != 0 {
                let message = if message.is_null() {
                    "Unknown OIDN error".into()
                } else {
                    unsafe { CStr::from_ptr(message) }
                        .to_string_lossy()
                        .into_owned()
                };
                bail!("OIDN error {code}: {message}");
            }
            Ok(())
        }
        fn clear_images(&mut self) {
            unsafe {
                for filter in self.filters.drain(..) {
                    (self.api.oidnReleaseFilter)(filter);
                }
                for buffer in self.buffers.drain(..) {
                    (self.api.oidnReleaseBuffer)(buffer);
                }
            }
            self.size = [0; 2];
        }
        fn configure(&mut self, size: [u32; 2]) -> Result<()> {
            if self.size == size {
                return Ok(());
            }
            self.clear_images();
            let n = size[0] as usize * size[1] as usize;
            ensure!(n > 0, "OIDN requires a nonempty image");
            for bytes in [n * size_of::<super::super::Input>(), n * 12, n * 12] {
                let buffer = unsafe { (self.api.oidnNewBuffer)(self.device, bytes) };
                self.check()?;
                ensure!(!buffer.is_null(), "OIDN buffer allocation failed");
                self.buffers.push(buffer);
            }
            for world in 0..2 {
                for kind in 0..3 {
                    let filter = unsafe { (self.api.oidnNewFilter)(self.device, c"RT".as_ptr()) };
                    self.check()?;
                    ensure!(!filter.is_null(), "OIDN filter allocation failed");
                    self.filters.push(filter);
                    let image = |name: &CStr, buffer: Handle, offset: usize, stride: usize| unsafe {
                        (self.api.oidnSetFilterImage)(
                            filter,
                            name.as_ptr(),
                            buffer,
                            3,
                            size[0] as usize,
                            size[1] as usize,
                            offset,
                            stride,
                            0,
                        );
                    };
                    let base = world * 48;
                    let stride = size_of::<super::super::Input>();
                    match kind {
                        0 => {
                            image(c"color", self.buffers[1], 0, 12);
                            image(c"normal", self.buffers[0], base + 16, stride);
                            image(c"albedo", self.buffers[0], base + 32, stride);
                            image(c"output", self.buffers[2], 0, 12);
                            unsafe {
                                (self.api.oidnSetFilterBool)(filter, c"hdr".as_ptr(), true);
                                (self.api.oidnSetFilterBool)(filter, c"cleanAux".as_ptr(), true);
                            }
                        }
                        1 => {
                            image(c"albedo", self.buffers[0], base + 32, stride);
                            image(c"output", self.buffers[0], base + 32, stride);
                        }
                        _ => {
                            image(c"normal", self.buffers[0], base + 16, stride);
                            image(c"output", self.buffers[0], base + 16, stride);
                        }
                    }
                    unsafe {
                        (self.api.oidnSetFilterInt)(filter, c"quality".as_ptr(), 6);
                        (self.api.oidnCommitFilter)(filter);
                    }
                    self.check()?;
                }
            }
            self.size = size;
            Ok(())
        }
        #[cfg(test)]
        pub fn filter(
            &mut self,
            size: [u32; 2],
            input: &[super::super::Input],
            world: usize,
        ) -> Result<Vec<[f32; 3]>> {
            Ok(self.reconstruct(size, input, world, false)?.color)
        }
        pub fn reconstruct(
            &mut self,
            size: [u32; 2],
            input: &[super::super::Input],
            world: usize,
            uncertainty: bool,
        ) -> Result<super::Reconstruction> {
            ensure!(
                world < 2 && input.len() == size[0] as usize * size[1] as usize,
                "Invalid OIDN image shape"
            );
            self.configure(size)?;
            unsafe {
                (self.api.oidnWriteBuffer)(
                    self.buffers[0],
                    0,
                    std::mem::size_of_val(input),
                    input.as_ptr().cast(),
                );
                (self.api.oidnExecuteFilter)(self.filters[world * 3 + 1]);
                (self.api.oidnExecuteFilter)(self.filters[world * 3 + 2]);
            }
            self.check()?;
            let render = |half: usize| -> Result<Vec<[f32; 3]>> {
                let color: Vec<[f32; 3]> = input
                    .iter()
                    .map(|p| {
                        let data = match (world, half) {
                            (0, 0) => p.color_a,
                            (1, 0) => p.color_b,
                            (0, 1) => p.half_a0,
                            (0, _) => p.half_a1,
                            (1, 1) => p.half_b0,
                            _ => p.half_b1,
                        };
                        [data[0].max(0.), data[1].max(0.), data[2].max(0.)]
                    })
                    .collect();
                unsafe {
                    (self.api.oidnWriteBuffer)(
                        self.buffers[1],
                        0,
                        std::mem::size_of_val(color.as_slice()),
                        color.as_ptr().cast(),
                    );
                    (self.api.oidnExecuteFilter)(self.filters[world * 3]);
                }
                self.check()?;
                let mut output = vec![[0_f32; 3]; input.len()];
                unsafe {
                    (self.api.oidnReadBuffer)(
                        self.buffers[2],
                        0,
                        std::mem::size_of_val(output.as_slice()),
                        output.as_mut_ptr().cast(),
                    );
                }
                self.check()?;
                ensure!(
                    output.iter().flatten().all(|v| v.is_finite()),
                    "OIDN returned non-finite radiance"
                );
                Ok(output)
            };
            let color = render(0)?;
            let halves = if uncertainty {
                Some([render(1)?, render(2)?])
            } else {
                None
            };
            Ok(super::Reconstruction { color, halves })
        }
    }
    impl Drop for Engine {
        fn drop(&mut self) {
            self.clear_images();
            if !self.device.is_null() {
                unsafe {
                    (self.api.oidnReleaseDevice)(self.device);
                }
            }
        }
    }
}
pub(super) use symbols::Engine;
