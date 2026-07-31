//! # wgpu-info
//!
//! Portable querying of GPU capabilities through [`wgpu`], so applications can stop
//! guessing conservative limits. The classic symptom this solves: code that caps itself
//! at 8 storage buffers because it builds the device with [`wgpu::Limits::default`] and
//! never asks the adapter what it actually supports.
//!
//! ## Quick start
//!
//! ```no_run
//! let report = gpu_info::query();
//! for adapter in &report.adapters {
//!     println!(
//!         "{} ({}): max_storage_buffers_per_shader_stage = {}",
//!         adapter.name, adapter.backend,
//!         adapter.limits.max_storage_buffers_per_shader_stage,
//!     );
//! }
//! ```
//!
//! ## Migrating away from `Limits::default()`
//!
//! Replace the conservative default with the adapter's real maximums:
//!
//! ```no_run
//! # async fn demo(adapter: &wgpu::Adapter) -> Result<(), wgpu::RequestDeviceError> {
//! // Before: required_limits: wgpu::Limits::default()   // caps at 8 storage buffers
//! let (device, queue) = gpu_info::request_max_device(adapter, wgpu::Features::empty()).await?;
//! # let _ = (device, queue); Ok(())
//! # }
//! ```

/// Canonical GPU media-image handle ([`GpuImage`]) shared by every cluster GPU consumer.
/// Lives here because this crate anchors wgpu and owns [`shared_device`] (cycle-free).
pub mod image;
mod model;
mod vram;
/// Windows RAM via `GlobalMemoryStatusEx` (a syscall, not a `wmic` process spawn) — see [`win_mem`].
#[cfg(windows)]
mod win_mem;

/// OS-level VRAM + system RAM query without a GPU context, no wgpu: `nvidia-smi` / `reg query` /
/// sysfs / `system_profiler`. `os` itself stays `#![forbid(unsafe_code)]`; the ONE exception is
/// Windows RAM, which uses the `GlobalMemoryStatusEx` SYSCALL (isolated in [`win_mem`]) instead of a
/// `wmic` process spawn — that spawn hitched a consumer's UI thread on every poll.
/// Complements the wgpu capability report and the DXGI [`vram`] adapter budget.
pub mod os;

pub use image::{GpuImage, GpuImageError};
pub use model::{AdapterReport, DownlevelReport, GpuReport, TextureFormatReport};
pub use vram::{
    GpuVramContext, VramInfo, VramQuerier, vram_budget_bytes, vram_budget_from_context,
};

/// Re-exported so consumers spell `wgpu` types from a single, version-matched source.
pub use wgpu;

/// Enumerate every adapter wgpu can find across all backends and report its capabilities.
///
/// Synchronous: wgpu's adapter enumeration is async internally and is driven to completion
/// here via [`pollster`].
#[must_use]
pub fn query() -> GpuReport {
    query_backends(wgpu::Backends::all())
}

/// Like [`query`], but restricted to the given backends (e.g. `wgpu::Backends::VULKAN`).
#[must_use]
pub fn query_backends(backends: wgpu::Backends) -> GpuReport {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));

    GpuReport {
        wgpu_info_version: env!("CARGO_PKG_VERSION").to_string(),
        wgpu_version: WGPU_VERSION.to_string(),
        backends_requested: backends
            .iter_names()
            .map(|(name, _)| name.to_ascii_lowercase())
            .collect(),
        adapters: adapters.iter().map(AdapterReport::from_adapter).collect(),
    }
}

/// The limits an application should request to use the adapter to its fullest.
///
/// This is exactly [`wgpu::Adapter::limits`]; the named helper documents intent at the call
/// site — pass the result as `required_limits` instead of [`wgpu::Limits::default`].
#[must_use]
pub fn recommended_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    adapter.limits()
}

/// Supported MSAA sample counts for `format` on this adapter (always includes `1`), e.g. `[1, 4]`.
///
/// Cross-platform: the answer comes straight from `wgpu`'s backend-reported
/// [`wgpu::TextureFormatFeatureFlags`], so it is accurate on Metal / Vulkan / DX12 / GL alike
/// (no platform-specific code, no hard-coded `4`). Renderers should intersect the result across
/// EVERY attachment format a pass uses — e.g. the color target AND the depth format — because a
/// GPU may support more MSAA counts for color than for depth; committing to a level only one
/// side supports panics at texture creation.
#[must_use]
pub fn supported_sample_counts(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> Vec<u32> {
    model::sample_counts_from_flags(adapter.get_texture_format_features(format).flags)
}

/// A compact, multi-line summary of a chosen adapter for startup logging.
#[must_use]
pub fn adapter_summary(adapter: &wgpu::Adapter) -> String {
    use core::fmt::Write as _;
    let r = AdapterReport::from_adapter(adapter);
    let l = &r.limits;
    let vendor = r
        .vendor_name
        .as_deref()
        .map(|v| format!("{v} (0x{:04x})", r.vendor & 0xffff))
        .unwrap_or_else(|| format!("0x{:04x}", r.vendor & 0xffff));
    let mut s = String::new();
    let _ = writeln!(s, "GPU: {}  [{}, {}]", r.name, r.backend, r.device_type);
    let _ = writeln!(
        s,
        "  vendor   : {vendor}   device 0x{:04x}",
        r.device & 0xffff
    );
    if !r.driver.is_empty() || !r.driver_info.is_empty() {
        let _ = writeln!(s, "  driver   : {} {}", r.driver, r.driver_info);
    }
    let _ = writeln!(
        s,
        "  shader   : model={}  webgpu_compliant={}  subgroup={}..{}",
        r.downlevel.shader_model,
        r.downlevel.is_webgpu_compliant,
        r.subgroup_min_size,
        r.subgroup_max_size
    );
    let _ = writeln!(
        s,
        "  buffers  : storage/stage={}  uniform/stage={}  max_storage_binding={}  max_buffer={}",
        l.max_storage_buffers_per_shader_stage,
        l.max_uniform_buffers_per_shader_stage,
        l.max_storage_buffer_binding_size,
        l.max_buffer_size
    );
    let _ = writeln!(
        s,
        "  compute  : invocations/wg={}  wg_size=[{}, {}, {}]  wg/dim={}",
        l.max_compute_invocations_per_workgroup,
        l.max_compute_workgroup_size_x,
        l.max_compute_workgroup_size_y,
        l.max_compute_workgroup_size_z,
        l.max_compute_workgroups_per_dimension
    );
    let _ = write!(
        s,
        "  binding  : bind_groups={}  sampled_textures/stage={}  samplers/stage={}",
        l.max_bind_groups, l.max_sampled_textures_per_shader_stage, l.max_samplers_per_shader_stage
    );
    s
}

/// Request a device with the adapter's **maximum** limits, plus any extra features.
///
/// `extra_features` is intersected with what the adapter supports. NOTE (wgpu 30): do NOT pass
/// [`wgpu::Features::all`] — the adapter reports EXPERIMENTAL features (ray-query, mesh-shader,
/// cooperative-matrix, …) that `request_device` REJECTS unless you flip `unsafe`
/// `ExperimentalFeatures::enabled()`. Request every *stable* feature with
/// `wgpu::Features::all() & !wgpu::Features::all_experimental_mask()` instead (this is what
/// [`shared_device`] does). The returned device's limits equal `adapter.limits()`, eliminating the
/// `Limits::default()` guessing game.
pub async fn request_max_device(
    adapter: &wgpu::Adapter,
    extra_features: wgpu::Features,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu-info max-limits device"),
            required_features: extra_features & adapter.features(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
}

/// Blocking wrapper around [`request_max_device`] for non-async callers.
pub fn request_max_device_blocking(
    adapter: &wgpu::Adapter,
    extra_features: wgpu::Features,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    pollster::block_on(request_max_device(adapter, extra_features))
}

/// The process-wide shared GPU context: ONE negotiation, ONE physical device.
///
/// `wgpu`'s [`Device`](wgpu::Device), [`Queue`](wgpu::Queue) and [`Adapter`](wgpu::Adapter) are
/// cheap `Arc`-backed handles, so a consumer `.clone()`s these fields into its own device-adoption
/// entry point (e.g. jph-wgpu's `Gpu::from_parts`, exv-gpu's `from_wgpu`) and every consumer then
/// shares ONE physical device. That is what makes zero-copy interop possible: decoded frames and
/// compute results all live on the same device instead of being split across independently
/// negotiated ones.
pub struct SharedGpu {
    /// The shared logical device (max limits + every supported feature). Clone to adopt.
    pub device: wgpu::Device,
    /// The queue paired with [`device`](Self::device). Clone to adopt.
    pub queue: wgpu::Queue,
    /// The adapter the device was created from. Clone to adopt (some adopters want the adapter).
    pub adapter: wgpu::Adapter,
    /// The instance the adapter came from. Clone to adopt.
    ///
    /// Needed by GUI adopters: `egui_wgpu::WgpuSetupExisting` wants all four handles, and a
    /// surface must be created from the SAME instance the adapter belongs to. Without this field
    /// an egui/eframe host had to negotiate its own instance+adapter+device, which put the UI on a
    /// different physical device than every compute consumer adopting this one — exactly the split
    /// this module exists to prevent.
    pub instance: wgpu::Instance,
}

/// Cached result of THE single process-wide device negotiation. `None` = the negotiation ran and
/// no adapter/device was available — a cached negative, so a GPU-less machine does not re-probe
/// on every call.
static SHARED: std::sync::OnceLock<Option<SharedGpu>> = std::sync::OnceLock::new();

/// Borrow the process-wide shared GPU context, negotiating it EXACTLY ONCE.
///
/// This is THE single `Instance` / `request_adapter` / `request_device` negotiation for the whole
/// process. Every GPU consumer in the cluster should ADOPT this context — cloning the handles into
/// its own `from_parts` / `from_wgpu` adopter — rather than creating its own [`wgpu::Instance`].
/// Independent negotiations were the historical cause of test-suite hangs and split decoded/compute
/// results across two physical devices.
///
/// The device is built with the adapter's MAXIMUM limits and every supported *stable* feature
/// ([`wgpu::Features::all`] minus [`wgpu::Features::all_experimental_mask`], intersected with the
/// adapter's real feature set), so an adopter that needs e.g. `TIMESTAMP_QUERY` or
/// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` for its fast path finds them present, while an
/// adopter that doesn't simply ignores them. Experimental features (ray-query, mesh-shader,
/// cooperative-matrix …) are excluded on purpose: wgpu 30 rejects them at `request_device` unless
/// the caller flips an `unsafe { ExperimentalFeatures::enabled() }` opt-in, and requesting them
/// would make the whole negotiation fail with "experimental features are not enabled".
///
/// Returns `None` (cached) when no adapter is available or device creation fails — never panics.
/// [`OnceLock::get_or_init`] collapses concurrent first callers into ONE negotiation.
pub fn shared_device() -> Option<&'static SharedGpu> {
    SHARED
        .get_or_init(|| {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            // HighPerformance + no fallback: adopt the real discrete GPU, not a software rasterizer.
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false, // wgpu 30
                }))
                .ok()?;
            // Every stable feature, minus experimental ones (which need an unsafe instance opt-in),
            // so ANY adopter's fast path is satisfied on the one shared device without the whole
            // negotiation failing. `request_max_device` intersects this with `adapter.features()`.
            let stable_features = wgpu::Features::all() & !wgpu::Features::all_experimental_mask();
            let (device, queue) =
                pollster::block_on(request_max_device(&adapter, stable_features)).ok()?;
            Some(SharedGpu {
                device,
                queue,
                adapter,
                instance,
            })
        })
        .as_ref()
}

/// Major wgpu version this crate targets. Pinned to the `wgpu = "30"` dependency in
/// `Cargo.toml`; semver keeps the major at 30 for every 30.x patch.
const WGPU_VERSION: &str = "30";

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU-free synthetic report for testing serialization and diffing.
    fn sample() -> GpuReport {
        GpuReport {
            wgpu_info_version: "0.1.0".into(),
            wgpu_version: "29".into(),
            backends_requested: vec!["vulkan".into()],
            adapters: vec![AdapterReport {
                name: "Test GPU".into(),
                backend: "vulkan".into(),
                device_type: "DiscreteGpu".into(),
                vendor: 0x10de,
                vendor_name: Some("NVIDIA".into()),
                device: 0x1234,
                pci_bus_id: "0000:01:00.0".into(),
                driver: "test".into(),
                driver_info: "1.0".into(),
                subgroup_min_size: 32,
                subgroup_max_size: 32,
                features: vec!["TEXTURE_BINDING_ARRAY".into()],
                limits: wgpu::Limits::default(),
                downlevel: DownlevelReport {
                    is_webgpu_compliant: true,
                    shader_model: "Sm5".into(),
                    flags: vec!["COMPUTE_SHADERS".into()],
                },
                texture_formats: vec![],
            }],
        }
    }

    /// MSAA flag -> concrete counts mapping is GPU-free (synthetic flags), so it runs anywhere.
    #[test]
    fn sample_counts_from_flags_maps_msaa() {
        use wgpu::TextureFormatFeatureFlags as F;
        // No MSAA flags -> only single-sampled.
        assert_eq!(crate::model::sample_counts_from_flags(F::empty()), vec![1]);
        // 2x + 4x present, 8x/16x absent.
        let counts = crate::model::sample_counts_from_flags(F::MULTISAMPLE_X2 | F::MULTISAMPLE_X4);
        assert_eq!(counts, vec![1, 2, 4]);
    }

    #[test]
    fn json_round_trip() {
        let report = sample();
        let json = serde_json::to_string(&report).unwrap();
        let back: GpuReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
    }

    #[test]
    fn diff_identical_is_empty() {
        assert!(sample().diff(&sample()).is_empty());
    }

    /// Requires a real GPU adapter; run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires a real GPU adapter"]
    fn live_query_reports_real_limits() {
        let report = query();
        assert!(
            !report.adapters.is_empty(),
            "no adapters enumerated on this system"
        );
        assert!(
            report
                .adapters
                .iter()
                .any(|a| a.limits.max_storage_buffers_per_shader_stage >= 8),
            "expected at least one adapter at or above the baseline storage-buffer limit"
        );
    }

    /// Pins the single-negotiation guarantee: two calls return the SAME `&SharedGpu`.
    /// Requires a real GPU adapter; run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires a GPU"]
    fn shared_device_is_singleton() {
        let a = shared_device().expect("no shared GPU device available on this system");
        let b = shared_device().expect("shared device vanished on the second call");
        assert!(
            std::ptr::eq(a, b),
            "shared_device() returned two distinct contexts — negotiation ran more than once"
        );
    }

    /// GPU-free OnceLock idempotency: repeated calls yield the same cached state (both `Some`
    /// or both `None`) and a pointer-stable `&'static`. Safe even without a GPU (returns `None`).
    #[test]
    fn shared_device_is_idempotent() {
        let first = shared_device();
        let second = shared_device();
        assert_eq!(
            first.is_some(),
            second.is_some(),
            "shared_device() Option-state changed between calls"
        );
        match (first, second) {
            (Some(a), Some(b)) => {
                assert!(std::ptr::eq(a, b), "cached context is not pointer-stable")
            }
            (None, None) => {}
            _ => unreachable!("is_some() equality already asserted above"),
        }
    }

    #[test]
    fn diff_detects_limit_change() {
        let a = sample();
        let mut b = sample();
        b.adapters[0].limits.max_storage_buffers_per_shader_stage = 64;
        let diff = a.diff(&b);
        assert_eq!(diff.len(), 1);
        assert!(diff[0].contains("maxStorageBuffersPerShaderStage"));
        assert!(diff[0].contains("8 -> 64"));
    }

    /// Calls `vram_budget_bytes` against live adapters; no panic.
    #[test]
    #[ignore = "requires a GPU"]
    fn live_vram_budget_no_panic() {
        let report = query();
        if report.adapters.is_empty() {
            return;
        }
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(desc);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        for adapter in &adapters {
            let budget = vram_budget_bytes(adapter);
            println!(
                "{}: vram_budget_bytes = {:?}",
                adapter.get_info().name,
                budget
            );
        }
    }
}
