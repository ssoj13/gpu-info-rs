//! GPU resource limits, and the one question every tiling consumer asks:
//! **can this image be handed to the device in a single dispatch, and if not, why?**
//!
//! # Why this lives here
//!
//! Two crates in the cluster grew their own answer to that question, independently:
//! `vfx_compute::backend::tiling::GpuLimits` and `vfx_warp::backend::tiling::ResourceLimits`.
//! They carried the same three facts (`max_tile_dim`, `max_buffer_bytes`, `available_memory`) and
//! **drifted in opposite directions**, each correct where the other was wrong — one clamped tile
//! size by the byte cap and took its buffer limit from a guess, the other did the reverse. Both
//! also shipped the same defect: `needs_tiling` consulted only the texture dimension while the
//! pipelines bind storage BUFFERS, which are capped in bytes.
//!
//! Neither crate depends on the other, so there was nowhere for the shared answer to live. There
//! is now: both already depend on this crate, and this crate already owns the neighbouring
//! knowledge — `shared_device`, `dedicated_vram`, `free_vram`, `vram_budget_bytes`.
//!
//! # FACTS here, POLICY in the consumer
//!
//! This module deliberately holds only what the DEVICE imposes. It does **not** decide tile sizes.
//! That is policy, it genuinely differs between consumers — a warper caps tiles at 4096 with no
//! safety margin; a compute pipeline rounds to a power of two with a floor of 256 and reserves 20%
//! of VRAM — and folding two policies into one would silently change behaviour for whichever
//! consumer lost. Margins and tile heuristics stay with the consumer that owns them; only the
//! measurements and the fit question are shared.
//!
//! # No wgpu in the core
//!
//! The type and every method below are wgpu-free, so a CPU-only or CUDA-only consumer can use them
//! without pulling a graphics stack. Only [`GpuLimits::from_wgpu_limits`] is behind the `wgpu`
//! feature.

/// Why a single dispatch cannot cover a whole image.
///
/// Named rather than a bare `bool` because the two causes have different remedies and thresholds
/// far apart — and because a caller that only sees `true` cannot tell an operator whether the
/// machine is short of memory or the image is simply too wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TilingReason {
    /// Wider or taller than [`GpuLimits::max_tile_dim`] — a TEXTURE dimension limit.
    TextureDimension,
    /// Does not fit one buffer binding ([`GpuLimits::max_buffer_bytes`]) — a BYTE limit.
    BufferBinding,
}

/// What the device can actually accept.
///
/// Every field is a measurement or an adapter-reported cap, never a heuristic. Construct with
/// [`GpuLimits::from_wgpu_limits`] when a wgpu adapter is at hand, or [`GpuLimits::new`] from
/// whatever the CUDA / CPU backend reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLimits {
    /// Largest texture dimension in either axis.
    pub max_tile_dim: u32,
    /// Largest single BUFFER BINDING, in bytes.
    ///
    /// For a wgpu consumer this is `max_storage_buffer_binding_size`, **not** `max_buffer_size`:
    /// a compute shader binding `var<storage> array<f32>` is capped by the former, which is the
    /// smaller of the two on typical adapters (wgpu's defaults are 128 MiB against 256 MiB).
    /// Sizing work by the larger number admits dispatches the bind then rejects.
    pub max_buffer_bytes: u64,
    /// Total device memory, UNMARGINED. A field named `total` must mean total; apply safety
    /// margins when computing [`Self::available_memory`], once, in the consumer that owns the policy.
    pub total_memory: u64,
    /// Memory a consumer may plan against, after whatever margin it applies.
    pub available_memory: u64,
    /// Were these measured, or are they fallback defaults? A planner that knows it is guessing
    /// can say so instead of presenting a default as a fact.
    pub detected: bool,
}

impl GpuLimits {
    /// Build from values the caller has already established.
    #[must_use]
    pub const fn new(
        max_tile_dim: u32,
        max_buffer_bytes: u64,
        total_memory: u64,
        available_memory: u64,
        detected: bool,
    ) -> Self {
        Self {
            max_tile_dim,
            max_buffer_bytes,
            total_memory,
            available_memory,
            detected,
        }
    }

    /// Limits for a CPU backend: no device caps, memory is whatever the host has.
    #[must_use]
    pub const fn cpu(available_memory: u64) -> Self {
        Self::new(u32::MAX, u64::MAX, available_memory, available_memory, true)
    }

    /// No limits at all — for analysis paths that must not be constrained by a real device.
    #[must_use]
    pub const fn unconstrained() -> Self {
        Self::new(u32::MAX, u64::MAX, u64::MAX, u64::MAX, false)
    }

    /// Build from a wgpu adapter's own reported caps.
    ///
    /// `available_memory` is the caller's business: it is what remains after the caller's safety
    /// margin, and that margin is policy this module deliberately does not own.
    ///
    /// Takes `max_storage_buffer_binding_size` for [`Self::max_buffer_bytes`] — see that field.
    #[cfg(feature = "wgpu")]
    #[must_use]
    pub fn from_wgpu_limits(
        limits: &wgpu::Limits,
        total_memory: u64,
        available_memory: u64,
    ) -> Self {
        Self::new(
            limits.max_texture_dimension_2d,
            u64::from(limits.max_storage_buffer_binding_size),
            total_memory,
            available_memory,
            true,
        )
    }

    /// Bytes an image of this shape occupies. The single place a shape becomes bytes.
    ///
    /// `bytes_per_pixel` rather than a channel count, because consumers disagree about what a
    /// pixel is: a warper is RGBA `f32` throughout (16), a compute pipeline is parameterised by
    /// channel count. Both agree on bytes.
    #[must_use]
    pub const fn image_bytes(width: u32, height: u32, bytes_per_pixel: u64) -> u64 {
        (width as u64) * (height as u64) * bytes_per_pixel
    }

    /// Why one dispatch cannot cover the whole image — `None` when it can.
    ///
    /// **Two independent caps.** The dimensional one is obvious and was the only one both
    /// consumers checked; the byte one is where a buffer pipeline actually fails. At RGBA `f32` a
    /// 4K image is 126.6 MiB and an 8K one is 506 MiB, so a 128 MiB binding is exceeded between
    /// them — while both remain far inside a 16384 px texture limit. An 8K image was therefore
    /// planned as a single pass and failed when the bind group was created.
    ///
    /// Free memory is deliberately NOT a reason here. It is a budget question, it decides *tiled
    /// versus streaming* rather than *can one dispatch hold this*, and the answer depends on the
    /// consumer's working-set model.
    #[must_use]
    pub const fn tiling_reason(
        &self,
        width: u32,
        height: u32,
        bytes_per_pixel: u64,
    ) -> Option<TilingReason> {
        if width > self.max_tile_dim || height > self.max_tile_dim {
            return Some(TilingReason::TextureDimension);
        }
        if Self::image_bytes(width, height, bytes_per_pixel) > self.max_buffer_bytes {
            return Some(TilingReason::BufferBinding);
        }
        None
    }

    /// Must this image be split before it can be dispatched?
    ///
    /// Thin wrapper over [`Self::tiling_reason`], which is the one that says WHY.
    #[must_use]
    pub const fn needs_tiling(&self, width: u32, height: u32, bytes_per_pixel: u64) -> bool {
        self.tiling_reason(width, height, bytes_per_pixel).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RGBA f32, the shape both consumers argue about.
    const RGBA_F32: u64 = 16;

    /// The BYTE wall is real and the dimension check cannot see it.
    ///
    /// Memory is deliberately abundant so this can only fail for the right reason.
    ///
    /// Reddening mutation: drop the `image_bytes > max_buffer_bytes` arm from `tiling_reason`.
    #[test]
    fn tiling_reason_sees_the_binding_wall_the_dimension_check_cannot() {
        let limits = GpuLimits::new(16384, 128 << 20, 24 << 30, 19 << 30, true);

        // 4K = 126.6 MiB: under a 128 MiB binding, and far under the dimension cap.
        assert_eq!(limits.tiling_reason(3840, 2160, RGBA_F32), None);

        // 8K = 506 MiB: still only 7680 px, so the dimension check alone says nothing.
        assert_eq!(
            limits.tiling_reason(7680, 4320, RGBA_F32),
            Some(TilingReason::BufferBinding)
        );

        // And the dimensional wall still reports itself, distinctly.
        assert_eq!(
            limits.tiling_reason(20000, 100, RGBA_F32),
            Some(TilingReason::TextureDimension)
        );
    }

    /// The dimension is checked FIRST, so an image over both caps reports the dimension.
    ///
    /// Not arbitrary: a caller that cannot make the image narrower cannot fix it by using less
    /// memory, so the harder constraint is the more useful thing to report.
    #[test]
    fn the_dimensional_wall_is_reported_when_both_apply() {
        let limits = GpuLimits::new(1024, 1 << 20, 8 << 30, 6 << 30, true);
        assert_eq!(
            limits.tiling_reason(4096, 4096, RGBA_F32),
            Some(TilingReason::TextureDimension)
        );
    }

    /// `cpu` and `unconstrained` never demand tiling, whatever they are asked.
    #[test]
    fn the_unlimited_constructors_never_tile() {
        for limits in [GpuLimits::cpu(8 << 30), GpuLimits::unconstrained()] {
            assert_eq!(limits.tiling_reason(100_000, 100_000, RGBA_F32), None);
            assert!(!limits.needs_tiling(100_000, 100_000, RGBA_F32));
        }
    }

    /// `total_memory` means total: no constructor here may pre-margin it.
    ///
    /// Reddening mutation: have any constructor store an already-reduced figure in `total_memory`.
    /// This is pinned because the defect it prevents shipped once: a consumer stored an
    /// 80%-margined value here and then took another 40% off it, arriving at 0.48 of an estimate
    /// behind a comment claiming 40%.
    #[test]
    fn total_means_total() {
        const TOTAL: u64 = 8 << 30;
        assert_eq!(GpuLimits::cpu(TOTAL).total_memory, TOTAL);
        assert_eq!(GpuLimits::new(0, 0, TOTAL, TOTAL / 2, true).total_memory, TOTAL);
    }

    #[test]
    fn image_bytes_is_the_one_size_fact() {
        assert_eq!(GpuLimits::image_bytes(3840, 2160, RGBA_F32), 132_710_400);
        assert_eq!(GpuLimits::image_bytes(2, 3, 4), 24);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn from_wgpu_limits_takes_the_storage_binding_not_max_buffer_size() {
        let limits = wgpu::Limits {
            max_buffer_size: 512 << 20,
            max_storage_buffer_binding_size: 128 << 20,
            ..wgpu::Limits::default()
        };
        let g = GpuLimits::from_wgpu_limits(&limits, 8 << 30, 6 << 30);
        assert_eq!(g.max_buffer_bytes, 128 << 20);
        assert!(g.detected);
    }
}
