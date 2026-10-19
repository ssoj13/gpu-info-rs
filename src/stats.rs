//! Live GPU counters cheap enough to poll at UI rates (1-10 Hz).
//!
//! # Why this exists next to [`crate::os`]
//!
//! [`os`](mod@crate::os) answers "what GPU is this and how much VRAM does it have" by shelling
//! out: `system_profiler` (~1 s on macOS), `nvidia-smi`, `reg query`. That is acceptable for a
//! one-shot capability probe at start-up and completely wrong for a monitor widget — a process
//! spawn per sample hitches the caller's UI thread.
//!
//! **Contract of this module: no process spawns, ever.** Every backend is a syscall, an IOKit
//! property read, or a sysfs file read, so a caller may poll [`query`] straight from its frame
//! loop. Anything that cannot be answered that cheaply is reported as `None` rather than
//! silently falling back to a spawn.
//!
//! # Backends
//!
//! | Platform | Utilisation | Memory in use | Source |
//! |---|---|---|---|
//! | macOS (Apple GPU) | yes | yes | IOKit `IOAccelerator` → `PerformanceStatistics` (same numbers Activity Monitor graphs), unprivileged |
//! | Linux (AMD/Intel) | yes | yes | DRM sysfs `gpu_busy_percent`, `mem_info_vram_*` |
//! | Windows (any vendor) | yes | yes | PDH `GPU Engine` counters + DXGI `QueryVideoMemoryInfo` — the pair Task Manager reads, so NVML and `nvidia-smi` are not needed |
//! | Linux (NVIDIA) | no | no | needs NVML; `nvidia-smi` would be a spawn, so it is deliberately not used here |
//!
//! Callers that need the missing pieces should fall back to [`crate::os::query`]
//! themselves, on their own slow path, and cache the result.

#[cfg(target_os = "macos")]
mod apple;
#[cfg(target_os = "linux")]
mod drm;
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod nvml;
#[cfg(target_os = "windows")]
mod windows;

/// Adapter name and memory with no process spawn, for [`crate::os`]'s Windows path.
///
/// Re-exported here so `os` never reaches into a platform submodule of its own accord.
#[cfg(target_os = "windows")]
pub(crate) fn windows_adapter_memory() -> Option<(String, u64, u64, u64, bool)> {
    windows::adapter_memory()
}

/// Fills in the fields Windows itself does not publish, where a vendor library can.
///
/// Enrichment, not a fallback: the base reading is complete on its own for every vendor, and
/// this only adds what PDH has no counter for.
#[cfg(target_os = "windows")]
fn enrich_nth(stats: &mut GpuStats, ordinal: usize) {
    let Some(name) = stats.name.as_deref() else {
        return;
    };
    let Some(n) = nvml::query_nth(name, ordinal) else {
        return;
    };
    stats.temp_c = n.temp_c;
    stats.power_w = n.power_w;
    stats.power_limit_w = n.power_limit_w;
    stats.fan_pct = n.fan_pct;
    stats.clock_core_mhz = n.clock_core_mhz;
    stats.clock_mem_mhz = n.clock_mem_mhz;
    stats.mem_bus_pct = n.mem_bus_pct;
    stats.driver = n.driver;
}

/// One live GPU reading.
///
/// Every field is optional because platforms disagree about what they expose, and a wrong
/// number is worse than an absent one: `None` means "this platform will not tell us", which
/// a UI should render as `—` rather than as zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuStats {
    /// GPU / SoC name when the driver publishes one (e.g. `"Apple M4 Pro"`).
    pub name: Option<String>,
    /// Device utilisation in percent, `0.0..=100.0`.
    pub util_pct: Option<f32>,
    /// GPU-resident bytes. On unified-memory parts this is the GPU's slice of system RAM.
    pub mem_used_bytes: Option<u64>,
    /// Total memory the GPU can address. Equals system RAM when [`unified`](Self::unified).
    pub mem_total_bytes: Option<u64>,
    /// System RAM the driver has committed for a *discrete* GPU (Windows WDDM
    /// `GPU Adapter Memory \ Shared Usage`). `None` on unified parts, where
    /// [`mem_used_bytes`](Self::mem_used_bytes) already is this counter, and on platforms that
    /// do not publish it (DRM sysfs, IOKit).
    ///
    /// # Why it exists
    ///
    /// Added 2026-10-12 for lgpt-rs. When a workload outgrows dedicated VRAM, the Windows
    /// driver does not fail the allocation: it silently pages the overflow to host RAM
    /// ("sysmem fallback"). Every access then crosses PCIe, kernels run tens of times slower,
    /// and utilisation still reads high, so `mem_used_bytes` alone (pinned near the card's
    /// size) cannot tell a full card from a spilling one. Measured there: pythia-160m training
    /// went from 0.2 s to 20 s per step with 9.5 GB in this counter. Callers compare it with a
    /// baseline taken before the work and stop or shrink the job when it grows.
    ///
    /// # Caveats (what can break)
    ///
    /// * It is the **adapter-wide** total over all processes, like Task Manager: a browser or
    ///   compositor holds a few hundred MiB here at idle. Compare deltas against a baseline,
    ///   never the absolute value against zero.
    /// * It counts every shared allocation, not only paged-out VRAM: staging/upload buffers a
    ///   program maps on purpose grow it too. A growth threshold must leave room for them.
    /// * Adding this public field to a struct that is not `#[non_exhaustive]` breaks any
    ///   downstream code that builds `GpuStats { .. }` with an exhaustive literal; use
    ///   `..Default::default()`.
    /// * PDH resolves the counter by its English name (`PdhAddEnglishCounterW`); if the counter
    ///   set is missing (old Windows, disabled performance counters) the whole PDH query fails
    ///   and every live field is `None`, this one included.
    pub mem_shared_bytes: Option<u64>,
    /// GPU shares system RAM (Apple Silicon, integrated GPUs).
    pub unified: bool,
    /// Core temperature in °C.
    ///
    /// The fields below come from a vendor library where one is present — see
    /// [`nvml`](self) on Windows and Linux. Windows publishes no thermal, clock, power or fan
    /// counter of its own, so they are `None` on an AMD or Intel GPU until ADL and IGCL are
    /// wired in the same shape.
    pub temp_c: Option<f32>,
    /// Board power draw in watts.
    pub power_w: Option<f32>,
    /// The power limit currently enforced, in watts.
    pub power_limit_w: Option<f32>,
    /// Fan duty cycle as a percentage of maximum — not RPM.
    pub fan_pct: Option<f32>,
    /// Graphics clock in MHz.
    pub clock_core_mhz: Option<u32>,
    /// Memory clock in MHz.
    pub clock_mem_mhz: Option<u32>,
    /// Memory-*controller* utilisation: how busy the bus was, not how full the VRAM is.
    pub mem_bus_pct: Option<f32>,
    /// Driver version, when the vendor library reports one.
    pub driver: Option<String>,
}

impl GpuStats {
    /// Used fraction in `0.0..=1.0`, or `None` when either side of the ratio is unknown.
    pub fn mem_frac(&self) -> Option<f32> {
        let used = self.mem_used_bytes?;
        let total = self.mem_total_bytes?;
        if total == 0 {
            return None;
        }
        Some((used as f32 / total as f32).clamp(0.0, 1.0))
    }

    /// Power draw as a fraction of the enforced limit, or `None` when either side is unknown.
    pub fn power_frac(&self) -> Option<f32> {
        let used = self.power_w?;
        let limit = self.power_limit_w?;
        (limit > 0.0).then(|| (used / limit).clamp(0.0, 1.0))
    }

    /// True when the reading carries no usable counter, so a caller can skip drawing a section.
    pub fn is_empty(&self) -> bool {
        self.util_pct.is_none() && self.mem_used_bytes.is_none()
    }
}

/// Reads the primary GPU's live counters. Returns `None` when no backend applies.
///
/// Cheap enough to call every frame; see the module docs for the per-platform sources.
pub fn query() -> Option<GpuStats> {
    query_all().into_iter().next()
}

/// Reads live counters for every GPU on the machine without spawning any processes.
///
/// * **Windows:** DXGI + PDH across all hardware adapters in a single PDH collection pass,
///   enriched per-adapter from NVML (`nvml.dll`) when present.
/// * **Linux:** DRM sysfs (`amdgpu` / `i915` / `xe`) plus NVML (`libnvidia-ml.so.1`) for
///   NVIDIA adapters.
/// * **macOS:** IOKit `IOAccelerator` performance statistics.
pub fn query_all() -> Vec<GpuStats> {
    #[cfg(target_os = "macos")]
    {
        apple::query().into_iter().collect()
    }
    #[cfg(target_os = "linux")]
    {
        let mut out: Vec<GpuStats> = nvml::query_all()
            .into_iter()
            .map(|(name, n)| GpuStats {
                name: Some(name),
                util_pct: n.util_pct,
                mem_used_bytes: n.mem_used_bytes,
                mem_total_bytes: n.mem_total_bytes,
                mem_shared_bytes: None,
                unified: false,
                temp_c: n.temp_c,
                power_w: n.power_w,
                power_limit_w: n.power_limit_w,
                fan_pct: n.fan_pct,
                clock_core_mhz: n.clock_core_mhz,
                clock_mem_mhz: n.clock_mem_mhz,
                mem_bus_pct: n.mem_bus_pct,
                driver: n.driver,
            })
            .collect();
        out.extend(drm::query_all());
        out
    }
    #[cfg(target_os = "windows")]
    {
        let mut list = windows::query_all();
        let mut seen_by_name: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for s in &mut list {
            let key = s.name.clone().unwrap_or_default().to_ascii_lowercase();
            let ord = *seen_by_name.entry(key).and_modify(|c| *c += 1).or_insert(0);
            enrich_nth(s, ord);
        }
        list
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_frac_needs_both_sides() {
        let mut s = GpuStats {
            mem_used_bytes: Some(2 << 30),
            ..Default::default()
        };
        assert_eq!(s.mem_frac(), None, "no total => no fraction");
        s.mem_total_bytes = Some(8 << 30);
        assert_eq!(s.mem_frac(), Some(0.25));
    }

    #[test]
    fn mem_frac_is_clamped_and_zero_safe() {
        let over = GpuStats {
            mem_used_bytes: Some(16),
            mem_total_bytes: Some(8),
            ..Default::default()
        };
        assert_eq!(over.mem_frac(), Some(1.0));

        let zero = GpuStats {
            mem_used_bytes: Some(1),
            mem_total_bytes: Some(0),
            ..Default::default()
        };
        assert_eq!(zero.mem_frac(), None);
    }

    #[test]
    fn empty_means_no_counters() {
        assert!(GpuStats::default().is_empty());
        assert!(
            !GpuStats {
                util_pct: Some(0.0),
                ..Default::default()
            }
            .is_empty(),
            "0% utilisation is a real reading, not an absent one"
        );
    }

    /// Not an assertion about hardware — it proves the call is side-effect free and, on a
    /// machine with a supported GPU, that the backend actually decodes something.
    #[test]
    fn smoke_query() {
        let stats = query();
        eprintln!("gpu_info::stats::query() = {stats:?}");
        if let Some(s) = stats {
            if let Some(u) = s.util_pct {
                assert!((0.0..=100.0).contains(&u), "utilisation out of range: {u}");
            }
            if let (Some(used), Some(total)) = (s.mem_used_bytes, s.mem_total_bytes) {
                assert!(used <= total, "used {used} > total {total}");
            }
        }
    }

    /// The whole point of the module: polling must not cost a process spawn. A thousand
    /// samples finishing well inside a second is only possible on the syscall path.
    #[test]
    fn polling_is_cheap() {
        if query().is_none() {
            eprintln!("no GPU backend on this host, skipping cost check");
            return;
        }
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = query();
        }
        let per_call = start.elapsed() / 1000;
        eprintln!("gpu_info::stats::query() = {per_call:?} per call");
        assert!(
            per_call < std::time::Duration::from_millis(5),
            "query() costs {per_call:?} per call - that is spawn territory, not a syscall"
        );
    }
}
