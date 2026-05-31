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
//! let report = wgpu_info::query();
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
//! let (device, queue) = wgpu_info::request_max_device(adapter, wgpu::Features::empty()).await?;
//! # let _ = (device, queue); Ok(())
//! # }
//! ```

mod model;

pub use model::{AdapterReport, DownlevelReport, GpuReport, TextureFormatReport};
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

/// Request a device with the adapter's **maximum** limits, plus any extra features.
///
/// `extra_features` is intersected with what the adapter supports, so passing
/// [`wgpu::Features::all`] is safe (you get every supported feature). The returned device's
/// limits equal `adapter.limits()`, eliminating the `Limits::default()` guessing game.
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

/// Major wgpu version this crate targets. Pinned to the `wgpu = "29"` dependency in
/// `Cargo.toml`; semver keeps the major at 29 for every 29.x patch.
const WGPU_VERSION: &str = "29";

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
}
