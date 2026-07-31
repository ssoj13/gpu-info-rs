//! Canonical GPU media-image handle for the whole cluster.
//!
//! [`GpuImage`] is the ONE texture handle every GPU media consumer agrees on — colour
//! (vfx-ocio), compute ops (vfx-compute), decoders (jph-wgpu / exv), and a display/encoder
//! path all pass images around as a `GpuImage` instead of a raw [`wgpu::Texture`] plus loose
//! `(width, height, format)` metadata. It lives in `gpu-info-rs` because this crate is the
//! wgpu version anchor AND the owner of the process-wide [`shared_device`]; putting the handle
//! here is cycle-free (every GPU crate already depends on `gpu-info-rs`).
//!
//! ## Why one canonical format
//!
//! The texture is ALWAYS [`wgpu::TextureFormat::Rgba16Float`] (v1): HDR-capable half-float that
//! matches the backends' RGBA16F resident textures (jph-wgpu `LevelCache`, exv VRAM tiles) and a
//! display path, so `GpuImage::adopt` can wrap those residents with zero copy and every consumer
//! can bind/sample without a format-negotiation dance. [`GpuImage::format`] is exposed for future
//! flexibility, but reads `Rgba16Float` today.
//!
//! ## Why the shared device
//!
//! Allocation ([`GpuImage::new`] / [`GpuImage::upload_rgba_f32`] / [`GpuImage::read_rgba_f32`])
//! goes through [`shared_device`] — the single process-wide negotiation. These paths NEVER open a
//! new [`wgpu::Instance`] or call `request_adapter`, preserving the single-negotiation invariant
//! that makes zero-copy interop between decoders and compute possible (one physical device).
//! [`GpuImage::adopt`] takes a texture the caller already produced on that shared device.
//!
//! ## Why f16 at the host boundary
//!
//! `Rgba16Float` stores half-floats, but hosts speak `f32`. [`GpuImage::upload_rgba_f32`] packs
//! `f32 -> half::f16` before `write_texture`, and [`GpuImage::read_rgba_f32`] unpacks `f16 -> f32`
//! after readback. The round-trip is therefore lossy to f16 precision (~2.4e-4 max abs error for
//! values in `[0, 1]`); see the `round_trip` test.

use half::f16;

use crate::shared_device;

/// The canonical texture format for every [`GpuImage`] in v1: HDR-capable half-float RGBA.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Usage flags the canonical handle is allocated with. Deliberately broad so ONE handle serves
/// every consumer: `TEXTURE_BINDING` (compute/render sampling), `STORAGE_BINDING` (compute writes
/// — `Rgba16Float` is a WebGPU-spec storage format, so no device feature is required),
/// `COPY_SRC`/`COPY_DST` (readback + host upload) and `RENDER_ATTACHMENT` (adopt-from-render / a
/// display path).
const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT);

/// Bytes per texel for `Rgba16Float`: 4 channels x 2 bytes (f16).
const BYTES_PER_PIXEL: u32 = 8;

/// Errors from the [`GpuImage`] allocation / host-boundary paths. Every fallible wgpu-30 call
/// (`poll`, `get_mapped_range`, buffer map) maps to a typed variant — the API never panics.
#[derive(Debug, thiserror::Error)]
pub enum GpuImageError {
    /// [`shared_device`] returned `None` (no adapter / device on this machine). Callers that only
    /// [`GpuImage::adopt`] an existing texture never hit this.
    #[error("no shared GPU device available (shared_device() returned None)")]
    NoDevice,
    /// A zero width or height was requested; an empty texture is invalid.
    #[error("zero-size image is invalid")]
    ZeroSize,
    /// The requested dimensions exceed the device's `max_texture_dimension_2d`.
    #[error("image {width}x{height} exceeds device max_texture_dimension_2d {max}")]
    TooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
        /// The device limit that was exceeded.
        max: u32,
    },
    /// The host `rgba` slice length did not equal `width * height * 4` (interleaved RGBA f32).
    #[error("rgba slice length {got} != expected width*height*4 = {expected}")]
    PixelCount {
        /// Expected float count (`width * height * 4`).
        expected: usize,
        /// Actual slice length received.
        got: usize,
    },
    /// A buffer `map_async` operation failed (channel closed or wgpu map error).
    #[error("buffer map failed: {0}")]
    Map(String),
    /// [`wgpu::Device::poll`] failed while waiting for the readback map to complete.
    #[error("device poll failed: {0}")]
    Poll(String),
}

/// The cluster's canonical, zero-copy-friendly GPU media-image handle (see module docs).
///
/// Cheap to move; the underlying [`wgpu::Texture`] is `Arc`-backed, so cloning the texture via
/// [`GpuImage::texture`] and re-wrapping with [`GpuImage::adopt`] shares the same GPU allocation.
pub struct GpuImage {
    tex: wgpu::Texture,
    width: u32,
    height: u32,
    // format is always `FORMAT` (Rgba16Float) for v1; `format()` exists for future flexibility.
}

impl GpuImage {
    /// Allocate an empty `width` x `height` `Rgba16Float` texture on the shared device.
    ///
    /// Errors: [`GpuImageError::NoDevice`] if no shared device, [`GpuImageError::ZeroSize`] on a
    /// zero dimension, [`GpuImageError::TooLarge`] beyond `max_texture_dimension_2d`.
    pub fn new(width: u32, height: u32) -> Result<Self, GpuImageError> {
        let gpu = shared_device().ok_or(GpuImageError::NoDevice)?;
        validate_dims(width, height, &gpu.device)?;
        let tex = make_texture(&gpu.device, width, height);
        Ok(Self { tex, width, height })
    }

    /// Wrap an externally-produced texture (e.g. a backend decoder's resident output) WITHOUT
    /// copying. The caller guarantees `tex` was created on the shared device (so downstream
    /// consumers see it on the one physical device); `width`/`height` are recorded as-is.
    #[must_use]
    pub fn adopt(tex: wgpu::Texture, width: u32, height: u32) -> Self {
        Self { tex, width, height }
    }

    /// Upload interleaved RGBA `f32` (`width * height * 4` floats) from the host into a fresh
    /// texture. This is the CPU->GPU boundary: values are packed `f32 -> f16` before upload.
    ///
    /// Errors as [`GpuImage::new`], plus [`GpuImageError::PixelCount`] if `rgba.len()` is wrong.
    pub fn upload_rgba_f32(width: u32, height: u32, rgba: &[f32]) -> Result<Self, GpuImageError> {
        let gpu = shared_device().ok_or(GpuImageError::NoDevice)?;
        validate_dims(width, height, &gpu.device)?;
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(GpuImageError::PixelCount {
                expected,
                got: rgba.len(),
            });
        }

        let tex = make_texture(&gpu.device, width, height);

        // Host f32 -> f16 pack. `write_texture` has no 256-byte row-alignment requirement (unlike
        // buffer copies), so a tightly-packed `width * BYTES_PER_PIXEL` stride is correct here.
        let halfs: Vec<f16> = rgba.iter().map(|&v| f16::from_f32(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&halfs);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * BYTES_PER_PIXEL),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // Flush the staged write so the handle is immediately usable by any consumer, even one
        // that never submits its own work before sampling.
        gpu.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());

        Ok(Self { tex, width, height })
    }

    /// Read the texture back to interleaved RGBA `f32` on the host (GPU->CPU boundary). Blocking.
    ///
    /// Copies to an aligned staging buffer (256-byte `bytes_per_row`), maps it, and unpacks
    /// `f16 -> f32`. Every fallible wgpu-30 step returns a typed [`GpuImageError`] — no panics.
    pub fn read_rgba_f32(&self) -> Result<Vec<f32>, GpuImageError> {
        let gpu = shared_device().ok_or(GpuImageError::NoDevice)?;
        let (width, height) = (self.width, self.height);

        // copy_texture_to_buffer requires each row aligned to 256 bytes; pad the staging stride.
        let unpadded = width * BYTES_PER_PIXEL;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuImage readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuImageError::Poll(e.to_string()))?;
        rx.recv()
            .map_err(|_| GpuImageError::Map("map callback channel closed".into()))?
            .map_err(|e| GpuImageError::Map(e.to_string()))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| GpuImageError::Map(e.to_string()))?; // wgpu 30: fallible

        // Un-pad rows and unpack f16 -> f32. Each row start is a multiple of 256 (2-byte aligned),
        // so the u8 -> f16 cast is sound.
        let row_floats = width as usize * 4;
        let mut out = vec![0f32; row_floats * height as usize];
        for y in 0..height as usize {
            let row_start = y * padded as usize;
            let row_bytes = &data[row_start..row_start + unpadded as usize];
            let row_halfs: &[f16] = bytemuck::cast_slice(row_bytes);
            let dst = &mut out[y * row_floats..(y + 1) * row_floats];
            for (d, h) in dst.iter_mut().zip(row_halfs.iter()) {
                *d = h.to_f32();
            }
        }
        drop(data);
        buffer.unmap();
        Ok(out)
    }

    /// Borrow the underlying texture (for bind-group entries, further copies, etc.).
    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.tex
    }

    /// Create a default [`wgpu::TextureView`] for binding. A fresh view each call (views are cheap).
    #[must_use]
    pub fn view(&self) -> wgpu::TextureView {
        self.tex
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Image width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The texel format — always [`FORMAT`] (`Rgba16Float`) in v1.
    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        FORMAT
    }
}

/// Reject zero and over-limit dimensions before touching the GPU.
fn validate_dims(width: u32, height: u32, device: &wgpu::Device) -> Result<(), GpuImageError> {
    if width == 0 || height == 0 {
        return Err(GpuImageError::ZeroSize);
    }
    let max = device.limits().max_texture_dimension_2d;
    if width > max || height > max {
        return Err(GpuImageError::TooLarge { width, height, max });
    }
    Ok(())
}

/// Allocate the canonical `Rgba16Float` texture with the shared [`USAGE`] flags.
fn make_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("GpuImage"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: USAGE,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host->GPU->host round-trip reconstructs the data within f16 precision.
    ///
    /// Dimensions 17x13 are deliberately non-aligned so the readback exercises the 256-byte
    /// row padding (unpadded stride 17*8 = 136 bytes -> padded to 256). Values live in `[0, 1]`,
    /// where f16 has ~2.4e-4 max abs quantization error; we assert < 1e-3 (comfortable margin).
    #[test]
    #[ignore = "requires GPU"]
    fn round_trip() {
        let (w, h) = (17u32, 13u32);
        let n = (w * h * 4) as usize;
        let src: Vec<f32> = (0..n).map(|i| (i % 101) as f32 / 100.0).collect();

        let img = GpuImage::upload_rgba_f32(w, h, &src).expect("upload");
        assert_eq!((img.width(), img.height()), (w, h));
        assert_eq!(img.format(), FORMAT);

        let back = img.read_rgba_f32().expect("readback");
        assert_eq!(back.len(), src.len());

        let max_err = src
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("GpuImage f16 round-trip max abs error = {max_err:e}");
        assert!(
            max_err < 1e-3,
            "f16 round-trip error {max_err} exceeds 1e-3 bound"
        );
    }

    /// `adopt` records dims and exposes texture/view/accessors correctly.
    #[test]
    #[ignore = "requires GPU"]
    fn adopt_reports_dims() {
        let (w, h) = (8u32, 4u32);
        let made = GpuImage::new(w, h).expect("new");
        // Clone the Arc-backed texture and re-wrap it: same GPU allocation, fresh handle.
        let adopted = GpuImage::adopt(made.texture().clone(), w, h);
        assert_eq!((adopted.width(), adopted.height()), (w, h));
        assert_eq!(adopted.format(), FORMAT);
        assert_eq!(adopted.texture().size().width, w);
        assert_eq!(adopted.texture().size().height, h);
        let _view = adopted.view();
    }

    /// `new` succeeds on the shared device, and repeated `new` calls reuse the SAME `&'static`
    /// shared context — no second negotiation.
    #[test]
    #[ignore = "requires GPU"]
    fn new_reuses_shared_device() {
        let a = shared_device().expect("shared device");
        let _img1 = GpuImage::new(16, 16).expect("first new");
        let _img2 = GpuImage::new(32, 32).expect("second new");
        let b = shared_device().expect("shared device still present");
        assert!(
            std::ptr::eq(a, b),
            "GpuImage::new triggered a second device negotiation"
        );
    }
}
