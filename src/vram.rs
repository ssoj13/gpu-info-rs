//! Live driver VRAM querying (used / budget), cross-platform.
//!
//! # Why this exists
//!
//! [`crate::query`] reports *static* adapter capabilities (limits, features).
//! It does not say how much video memory the driver currently has committed,
//! nor the budget the OS is willing to hand this process. The viewer status bar
//! wants that *live* number next to the app's own tracked allocations.
//!
//! # Design
//!
//! [`VramQuerier`] is built ONCE from [`GpuVramContext`] and caches the platform
//! handle so per-frame [`VramQuerier::query`] is a cheap driver poll.
//!
//! # Platform matrix
//!
//! - **Windows** — DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` (adapter name-match;
//!   backend-agnostic — works when wgpu uses Vulkan too). VERIFIED.
//! - **Linux** — Vulkan `VK_EXT_memory_budget` via `Adapter::as_hal`.
//! - **macOS** — Metal `MTLDevice` via `Device::as_hal` (wgpu-hal 29 exposes
//!   `raw_device()` on the HAL **Device**, not the Adapter). Apple Silicon reports
//!   a unified-memory working set, not discrete VRAM.
//! - **Other** — `new` returns `None`; callers show tracked-only memory.

/// Inputs for live VRAM queries. Windows/Linux use `adapter`; macOS Metal requires
/// `device` (the same wgpu device the app renders with).
pub struct GpuVramContext<'a> {
    pub adapter: &'a wgpu::Adapter,
    pub device: &'a wgpu::Device,
}

/// A live GPU memory snapshot from the driver, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramInfo {
    /// Currently committed local video memory, bytes.
    pub used: u64,
    /// Current local video-memory budget for this process, bytes.
    pub budget: u64,
}

// ===========================================================================
// Windows — DXGI (verified path)
// ===========================================================================
#[cfg(windows)]
mod imp {
    use super::{GpuVramContext, VramInfo};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
        IDXGIFactory4,
    };
    use windows::core::Interface;

    pub struct VramQuerier {
        adapter: IDXGIAdapter3,
    }

    fn match_dxgi_adapter(want: &str) -> Option<IDXGIAdapter3> {
        let factory: IDXGIFactory4 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.ok()?;

        let mut best: Option<IDXGIAdapter3> = None;
        let mut first_hw: Option<IDXGIAdapter3> = None;
        let mut i = 0u32;
        loop {
            let base = match unsafe { factory.EnumAdapters1(i) } {
                Ok(a) => a,
                Err(_) => break,
            };
            i += 1;
            let Ok(desc) = (unsafe { base.GetDesc1() }) else {
                continue;
            };
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            let Ok(adapter3) = base.cast::<IDXGIAdapter3>() else {
                continue;
            };
            if first_hw.is_none() {
                first_hw = Some(adapter3.clone());
            }
            let len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            if !want.is_empty() && name.contains(want) {
                best = Some(adapter3);
                break;
            }
        }

        best.or(first_hw)
    }

    fn query_local_segment(adapter: &IDXGIAdapter3) -> Option<VramInfo> {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
            .ok()?;
        Some(VramInfo {
            used: info.CurrentUsage,
            budget: info.Budget,
        })
    }

    impl VramQuerier {
        pub fn new(ctx: GpuVramContext<'_>) -> Option<Self> {
            let want = ctx.adapter.get_info().name;
            let adapter = match_dxgi_adapter(&want)?;
            Some(Self { adapter })
        }

        pub fn query(&self) -> Option<VramInfo> {
            query_local_segment(&self.adapter)
        }
    }

    pub(super) fn vram_budget_adapter(adapter: &wgpu::Adapter) -> Option<u64> {
        let want = adapter.get_info().name;
        let dxgi = match_dxgi_adapter(&want)?;
        query_local_segment(&dxgi).map(|v| v.budget)
    }

    pub(super) fn vram_budget_from_context(ctx: GpuVramContext<'_>) -> Option<u64> {
        vram_budget_adapter(ctx.adapter)
    }
}

// ===========================================================================
// Linux / Android / FreeBSD — Vulkan VK_EXT_memory_budget
// ===========================================================================
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod imp {
    use super::{GpuVramContext, VramInfo};
    use ash::vk;

    pub struct VramQuerier {
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
    }

    impl VramQuerier {
        pub fn new(ctx: GpuVramContext<'_>) -> Option<Self> {
            let hal_adapter = unsafe { ctx.adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
            let raw_phys = hal_adapter.raw_physical_device();
            let shared = hal_adapter.shared_instance();
            let instance = shared.raw_instance().clone();
            Some(Self {
                instance,
                physical_device: raw_phys,
            })
        }

        pub fn query(&self) -> Option<VramInfo> {
            unsafe {
                let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
                let mut props2 =
                    vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
                self.instance
                    .get_physical_device_memory_properties2(self.physical_device, &mut props2);

                let mem = props2.memory_properties;
                let mut best_heap = usize::MAX;
                let mut best_size = 0u64;
                for h in 0..(mem.memory_heap_count as usize) {
                    let heap = mem.memory_heaps[h];
                    if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                        && heap.size > best_size
                    {
                        best_size = heap.size;
                        best_heap = h;
                    }
                }
                if best_heap == usize::MAX {
                    return None;
                }
                let used = budget.heap_usage[best_heap];
                let bud = budget.heap_budget[best_heap];
                if used == 0 && bud == 0 {
                    return None;
                }
                Some(VramInfo { used, budget: bud })
            }
        }
    }

    pub(super) fn vram_budget_adapter(adapter: &wgpu::Adapter) -> Option<u64> {
        let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let phys = hal_adapter.raw_physical_device();
        let raw_instance = hal_adapter.shared_instance().raw_instance();
        let props = unsafe { raw_instance.get_physical_device_memory_properties(phys) };
        let count = props.memory_heap_count as usize;
        let total: u64 = props.memory_heaps[..count]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        if total == 0 { None } else { Some(total) }
    }

    pub(super) fn vram_budget_from_context(ctx: GpuVramContext<'_>) -> Option<u64> {
        vram_budget_adapter(ctx.adapter)
    }
}

// ===========================================================================
// macOS — Metal via wgpu Device (wgpu-hal 29)
// ===========================================================================
#[cfg(target_os = "macos")]
mod imp {
    use super::{GpuVramContext, VramInfo};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLDevice;

    pub struct VramQuerier {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
    }

    fn mtl_from_wgpu_device(
        device: &wgpu::Device,
    ) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }?;
        Some(hal.raw_device().clone())
    }

    impl VramQuerier {
        pub fn new(ctx: GpuVramContext<'_>) -> Option<Self> {
            let device = mtl_from_wgpu_device(ctx.device)?;
            Some(Self { device })
        }

        pub fn query(&self) -> Option<VramInfo> {
            Some(VramInfo {
                used: self.device.currentAllocatedSize() as u64,
                budget: self.device.recommendedMaxWorkingSetSize(),
            })
        }
    }

    pub(super) fn vram_budget_from_context(ctx: GpuVramContext<'_>) -> Option<u64> {
        let mtl = mtl_from_wgpu_device(ctx.device)?;
        Some(mtl.recommendedMaxWorkingSetSize())
    }
}

// ===========================================================================
// Fallback — any other platform
// ===========================================================================
#[cfg(not(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "macos"
)))]
mod imp {
    use super::{GpuVramContext, VramInfo};

    pub struct VramQuerier;

    impl VramQuerier {
        pub fn new(_ctx: GpuVramContext<'_>) -> Option<Self> {
            None
        }

        pub fn query(&self) -> Option<VramInfo> {
            None
        }
    }

    pub(super) fn vram_budget_adapter(_adapter: &wgpu::Adapter) -> Option<u64> {
        None
    }

    pub(super) fn vram_budget_from_context(_ctx: GpuVramContext<'_>) -> Option<u64> {
        None
    }
}

/// Cached handle that queries live driver VRAM for one GPU context.
pub struct VramQuerier(imp::VramQuerier);

impl VramQuerier {
    /// Build once from the live adapter + device. Returns `None` when this platform
    /// cannot report live VRAM.
    #[must_use]
    pub fn new(ctx: GpuVramContext<'_>) -> Option<Self> {
        imp::VramQuerier::new(ctx).map(Self)
    }

    /// Current used/budget bytes from the driver.
    #[must_use]
    pub fn query(&self) -> Option<VramInfo> {
        self.0.query()
    }
}

/// Static VRAM budget (hardware total or OS working-set cap) in bytes.
///
/// Prefer this after the wgpu device exists — on macOS Metal the budget comes from
/// the same `MTLDevice` wgpu uses (`Device::as_hal`).
#[must_use]
pub fn vram_budget_from_context(ctx: GpuVramContext<'_>) -> Option<u64> {
    imp::vram_budget_from_context(ctx)
}

/// Adapter-only static budget. Works on Windows (DXGI) and Linux (Vulkan heap sum).
/// On macOS Metal returns `None` — call [`vram_budget_from_context`] after device
/// creation.
#[must_use]
pub fn vram_budget_bytes(adapter: &wgpu::Adapter) -> Option<u64> {
    #[cfg(windows)]
    {
        return imp::vram_budget_adapter(adapter);
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    if adapter.get_info().backend == wgpu::Backend::Vulkan {
        return imp::vram_budget_adapter(adapter);
    }

    #[cfg(target_os = "macos")]
    {
        let _ = adapter;
        log::debug!(
            "vram_budget_bytes: macOS Metal requires a wgpu Device; use vram_budget_from_context"
        );
        return None;
    }

    #[cfg(not(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "macos"
    )))]
    {
        let _ = adapter;
        return None;
    }

    #[allow(unreachable_code)]
    {
        let _ = adapter;
        None
    }
}

/// A rate-limited reading of the OS VRAM budget for a consumer whose idle quota follows it (a resource pool's
/// `set_budget`).
///
/// **Why:** the budget changes while the process runs (other processes, the OS), so a pool's quota is queried again
/// from time to time, but a driver query per retire is a syscall storm. The reading is cached for `every`; a failed
/// one (the platform reports none) is cached the same way - never remembered as a permanent zero - and logged once
/// (`log::warn!`, prefixed with the watch's `label`). Process-wide statics are the intended use (`const fn new`).
///
/// **Where used:** `ofx-runtime`'s GPU renderer pools; `ofx-finish`'s retained jobs keep an equivalent copy.
pub struct VramWatch {
    label: &'static str,
    every: std::time::Duration,
    state: std::sync::Mutex<WatchState>,
}

/// The last reading of a [`VramWatch`].
struct WatchState {
    budget: Option<u64>,
    at: Option<std::time::Instant>,
    failure_logged: bool,
}

impl std::fmt::Debug for VramWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramWatch")
            .field("label", &self.label)
            .field("every", &self.every)
            .finish_non_exhaustive()
    }
}

impl VramWatch {
    /// A watch that queries at most once per `every`; `label` names the consumer in the one failure log line.
    pub const fn new(label: &'static str, every: std::time::Duration) -> Self {
        Self {
            label,
            every,
            state: std::sync::Mutex::new(WatchState {
                budget: None,
                at: None,
                failure_logged: false,
            }),
        }
    }

    /// The OS VRAM budget of the context's device in bytes, read again when the last reading is older than the
    /// interval; `None` while the platform reports none.
    pub fn budget(&self, ctx: GpuVramContext<'_>) -> Option<u64> {
        self.budget_with(|| vram_budget_from_context(ctx))
    }

    /// [`Self::budget`] with the query supplied (a test drives the rate limit and the failure path with it).
    pub fn budget_with(&self, query: impl FnOnce() -> Option<u64>) -> Option<u64> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.at.is_some_and(|at| at.elapsed() < self.every) {
            return state.budget;
        }
        state.budget = query();
        if state.budget.is_none() && !state.failure_logged {
            log::warn!(
                "{}: the OS reports no VRAM budget for the device; nothing is kept idle (retried every {} ms)",
                self.label,
                self.every.as_millis()
            );
            state.failure_logged = true;
        }
        state.at = Some(std::time::Instant::now());
        state.budget
    }
}

#[cfg(test)]
mod watch_tests {
    use super::VramWatch;
    use std::cell::Cell;
    use std::time::Duration;

    #[test]
    fn queries_are_rate_limited_and_fresh_after_the_interval() {
        let watch = VramWatch::new("test", Duration::from_millis(60));
        let calls = Cell::new(0);
        let query = |value: u64| {
            calls.set(calls.get() + 1);
            Some(value)
        };
        assert_eq!(watch.budget_with(|| query(10)), Some(10));
        // Inside the interval the cached reading answers and the query does not run.
        assert_eq!(watch.budget_with(|| query(20)), Some(10));
        assert_eq!(calls.get(), 1);
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(watch.budget_with(|| query(30)), Some(30));
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn a_failed_query_is_cached_then_retried_not_remembered_as_zero() {
        let watch = VramWatch::new("test", Duration::from_millis(60));
        assert_eq!(watch.budget_with(|| None), None);
        assert_eq!(
            watch.budget_with(|| Some(7)),
            None,
            "cached inside the interval"
        );
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(watch.budget_with(|| Some(7)), Some(7));
    }
}
