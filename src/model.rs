//! Plain, serializable description of a system's GPU capabilities.
//!
//! The report deliberately embeds [`wgpu::Limits`] verbatim (it already derives
//! `Serialize`/`Deserialize`) so the full, current limit set is reported with zero
//! hand-maintained mirroring — a new limit in a future wgpu shows up automatically.
//! Everything else (features, backend, downlevel flags, formats) is flattened into
//! friendly string forms so the JSON is readable and stable across consumers.

use serde::{Deserialize, Serialize};

/// Full capability snapshot for every adapter wgpu can enumerate on this system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuReport {
    /// Version of this crate that produced the report.
    pub wgpu_info_version: String,
    /// wgpu version the report was built against.
    pub wgpu_version: String,
    /// Backends that were requested during enumeration (e.g. `["vulkan", "dx12"]`).
    pub backends_requested: Vec<String>,
    /// One entry per enumerated adapter. A single physical GPU may appear more than
    /// once when it is exposed through several backends (e.g. Vulkan and DX12).
    pub adapters: Vec<AdapterReport>,
}

/// Capabilities of a single adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterReport {
    /// Adapter name as reported by the driver.
    pub name: String,
    /// Backend this adapter is exposed through (`vulkan`, `dx12`, `metal`, `gl`, ...).
    pub backend: String,
    /// Device class: `DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, `Cpu`, `Other`.
    pub device_type: String,
    /// Backend-specific vendor id (usually a PCI vendor id in the low 16 bits).
    pub vendor: u32,
    /// Human-readable vendor name when the id is recognized.
    pub vendor_name: Option<String>,
    /// Backend-specific device id.
    pub device: u32,
    /// PCI bus id (`bus:device.function`) when the backend provides it.
    pub pci_bus_id: String,
    /// Driver name.
    pub driver: String,
    /// Driver version / build info.
    pub driver_info: String,
    /// Smallest subgroup (wave/warp) size on this adapter.
    pub subgroup_min_size: u32,
    /// Largest subgroup (wave/warp) size on this adapter.
    pub subgroup_max_size: u32,
    /// Supported optional [`wgpu::Features`], as SCREAMING_SNAKE_CASE names.
    pub features: Vec<String>,
    /// The best limits the adapter can grant — feed these to `request_device` instead
    /// of [`wgpu::Limits::default`] to stop guessing (this is the "8 buffers" fix).
    pub limits: wgpu::Limits,
    /// Downlevel (sub-WebGPU) capability description.
    pub downlevel: DownlevelReport,
    /// Per-format usage support for a curated set of common texture formats.
    pub texture_formats: Vec<TextureFormatReport>,
}

/// Downlevel capabilities — how the platform diverges from the baseline WebGPU spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DownlevelReport {
    /// True if the adapter fully meets the baseline WebGPU standard.
    pub is_webgpu_compliant: bool,
    /// Highest supported shader model (`Sm2`, `Sm4`, `Sm5`).
    pub shader_model: String,
    /// Set downlevel flags, as SCREAMING_SNAKE_CASE names.
    pub flags: Vec<String>,
}

/// Allowed usages and feature flags for one texture format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureFormatReport {
    /// Format name (e.g. `Rgba8Unorm`).
    pub format: String,
    /// Allowed [`wgpu::TextureUsages`] for this format, as flag names.
    pub allowed_usages: Vec<String>,
    /// Additional [`wgpu::TextureFormatFeatureFlags`], as flag names.
    pub flags: Vec<String>,
}

/// Texture formats probed by default — the common render/compute/depth/compressed set.
pub(crate) const COMMON_FORMATS: &[wgpu::TextureFormat] = {
    use wgpu::TextureFormat::*;
    &[
        R8Unorm, Rg8Unorm, Rgba8Unorm, Rgba8UnormSrgb, Bgra8Unorm, Bgra8UnormSrgb,
        Rgb10a2Unorm, Rg11b10Ufloat, R16Float, Rg16Float, Rgba16Float, R32Float,
        Rg32Float, Rgba32Float, R32Uint, Rgba32Uint, Depth16Unorm, Depth24Plus,
        Depth24PlusStencil8, Depth32Float, Bc1RgbaUnorm, Bc7RgbaUnorm,
    ]
};

/// Map a backend-specific vendor id to a human name.
///
/// The GL backend can report Mesa's full 17-bit id (`0x10005`); everything else uses the
/// low 16 bits as a PCI vendor id, so the Mesa case is matched on the unmasked value first.
pub(crate) fn vendor_name(vendor: u32) -> Option<&'static str> {
    if vendor == 0x10005 {
        return Some("Mesa");
    }
    match vendor & 0xffff {
        0x10de => Some("NVIDIA"),
        0x1002 | 0x1022 => Some("AMD"),
        0x8086 => Some("Intel"),
        0x13b5 => Some("ARM"),
        0x5143 => Some("Qualcomm"),
        0x1010 => Some("ImgTec"),
        0x106b => Some("Apple"),
        0x1414 => Some("Microsoft"), // WARP / software
        _ => None,
    }
}

fn device_type_str(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::Other => "Other",
        wgpu::DeviceType::IntegratedGpu => "IntegratedGpu",
        wgpu::DeviceType::DiscreteGpu => "DiscreteGpu",
        wgpu::DeviceType::VirtualGpu => "VirtualGpu",
        wgpu::DeviceType::Cpu => "Cpu",
    }
}

fn shader_model_str(m: wgpu::ShaderModel) -> &'static str {
    match m {
        wgpu::ShaderModel::Sm2 => "Sm2",
        wgpu::ShaderModel::Sm4 => "Sm4",
        wgpu::ShaderModel::Sm5 => "Sm5",
    }
}

impl AdapterReport {
    /// Build a report from a live [`wgpu::Adapter`].
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let downlevel = adapter.get_downlevel_capabilities();
        let texture_formats = COMMON_FORMATS
            .iter()
            .map(|&format| {
                let tf = adapter.get_texture_format_features(format);
                TextureFormatReport {
                    format: format!("{format:?}"),
                    allowed_usages: tf.allowed_usages.iter_names().map(|(n, _)| n.to_string()).collect(),
                    flags: tf.flags.iter_names().map(|(n, _)| n.to_string()).collect(),
                }
            })
            .collect();

        Self {
            name: info.name,
            backend: info.backend.to_str().to_string(),
            device_type: device_type_str(info.device_type).to_string(),
            vendor: info.vendor,
            vendor_name: vendor_name(info.vendor).map(str::to_string),
            device: info.device,
            pci_bus_id: info.device_pci_bus_id,
            driver: info.driver,
            driver_info: info.driver_info,
            subgroup_min_size: info.subgroup_min_size,
            subgroup_max_size: info.subgroup_max_size,
            features: adapter.features().iter_names().map(|(n, _)| n.to_string()).collect(),
            limits: adapter.limits(),
            downlevel: DownlevelReport {
                is_webgpu_compliant: downlevel.is_webgpu_compliant(),
                shader_model: shader_model_str(downlevel.shader_model).to_string(),
                flags: downlevel.flags.iter_names().map(|(n, _)| n.to_string()).collect(),
            },
            texture_formats,
        }
    }
}

impl GpuReport {
    /// Plain `(name, backend)` strings for the report's primary adapter.
    ///
    /// Why: UI consumers (e.g. the viewer status bar) want a one-line GPU
    /// label without spelling any `wgpu` types — keeping the wgpu-version
    /// boundary clean (this crate may compile against a different wgpu patch
    /// than the host). Returns owned `String`s copied out of [`AdapterReport`].
    ///
    /// Selection: a discrete GPU is preferred over integrated/CPU/other when
    /// several adapters are enumerated (the same physical GPU often appears
    /// once per backend); falls back to the first adapter, and to
    /// `("unknown", "none")` when no adapter was found.
    #[must_use]
    pub fn primary_summary(&self) -> (String, String) {
        let pick = self
            .adapters
            .iter()
            .find(|a| a.device_type == "DiscreteGpu")
            .or_else(|| self.adapters.first());
        match pick {
            Some(a) => (a.name.clone(), a.backend.clone()),
            None => ("unknown".to_string(), "none".to_string()),
        }
    }

    /// Compare two reports and return a list of human-readable differences
    /// (`path: old -> new`). Empty when the reports are identical.
    ///
    /// The diff walks the serialized JSON so it covers every field — including every
    /// limit — without per-field code. Adapters are compared positionally by index, so a
    /// stable enumeration order (same machine, same backends) is assumed; reordering
    /// between the two reports will surface as per-field differences.
    pub fn diff(&self, other: &GpuReport) -> Vec<String> {
        let a = serde_json::to_value(self).expect("report is serializable");
        let b = serde_json::to_value(other).expect("report is serializable");
        let mut out = Vec::new();
        diff_value("", &a, &b, &mut out);
        out
    }
}

fn diff_value(path: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            let mut keys: Vec<&String> = ma.keys().chain(mb.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                let child = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match (ma.get(k), mb.get(k)) {
                    (Some(va), Some(vb)) => diff_value(&child, va, vb, out),
                    (Some(va), None) => out.push(format!("{child}: {va} -> (removed)")),
                    (None, Some(vb)) => out.push(format!("{child}: (added) -> {vb}")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(aa), Value::Array(ab)) => {
            let n = aa.len().max(ab.len());
            for i in 0..n {
                let child = format!("{path}[{i}]");
                match (aa.get(i), ab.get(i)) {
                    (Some(va), Some(vb)) => diff_value(&child, va, vb, out),
                    (Some(va), None) => out.push(format!("{child}: {va} -> (removed)")),
                    (None, Some(vb)) => out.push(format!("{child}: (added) -> {vb}")),
                    (None, None) => {}
                }
            }
        }
        _ => {
            if a != b {
                out.push(format!("{path}: {a} -> {b}"));
            }
        }
    }
}

impl GpuReport {
    /// Render a human-readable, multi-adapter capability report.
    pub fn to_pretty(&self) -> String {
        use core::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "wgpu-info {} (wgpu {}) — backends: {}",
            self.wgpu_info_version,
            self.wgpu_version,
            self.backends_requested.join(", ")
        );
        if self.adapters.is_empty() {
            let _ = writeln!(s, "\nNo adapters found.");
            return s;
        }
        for (i, a) in self.adapters.iter().enumerate() {
            let _ = writeln!(s, "\n[{i}] {} ({})", a.name, a.backend);
            let vendor = a
                .vendor_name
                .as_deref()
                .map(|v| format!("{v} (0x{:04x})", a.vendor & 0xffff))
                .unwrap_or_else(|| format!("0x{:04x}", a.vendor & 0xffff));
            let _ = writeln!(s, "    type        : {}", a.device_type);
            let _ = writeln!(s, "    vendor      : {vendor}");
            let _ = writeln!(s, "    device      : 0x{:04x}", a.device & 0xffff);
            if !a.driver.is_empty() || !a.driver_info.is_empty() {
                let _ = writeln!(s, "    driver      : {} {}", a.driver, a.driver_info);
            }
            if !a.pci_bus_id.is_empty() {
                let _ = writeln!(s, "    pci bus id  : {}", a.pci_bus_id);
            }
            let _ = writeln!(s, "    subgroup    : {}..{}", a.subgroup_min_size, a.subgroup_max_size);
            let _ = writeln!(
                s,
                "    downlevel   : shader_model={}, webgpu_compliant={}",
                a.downlevel.shader_model, a.downlevel.is_webgpu_compliant
            );

            let _ = writeln!(s, "    limits:");
            for (k, v) in limits_kv(&a.limits) {
                let _ = writeln!(s, "        {k:<55} {v}");
            }

            let _ = writeln!(s, "    features ({}):", a.features.len());
            if a.features.is_empty() {
                let _ = writeln!(s, "        (none)");
            } else {
                for f in &a.features {
                    let _ = writeln!(s, "        {f}");
                }
            }
        }
        s
    }
}

/// Flatten [`wgpu::Limits`] into sorted `(name, value)` pairs via its serde form,
/// so every limit is printed without hand-listing fields.
fn limits_kv(limits: &wgpu::Limits) -> Vec<(String, String)> {
    let value = serde_json::to_value(limits).expect("Limits is serializable");
    let mut pairs: Vec<(String, String)> = match value {
        serde_json::Value::Object(map) => map
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect(),
        _ => Vec::new(),
    };
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}
