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
            Self::of(ctx.adapter)
        }

        /// The querier of `adapter` (Vulkan only).
        pub fn of(adapter: &wgpu::Adapter) -> Option<Self> {
            let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
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
                if bud == 0 {
                    return None;
                }
                Some(VramInfo { used, budget: bud })
            }
        }
    }

    /// The LIVE budget (`VK_EXT_memory_budget`) of the largest device-local heap, the same semantics as Windows'
    /// DXGI budget: what other processes use is already subtracted. `None` when the driver reports no budget (the
    /// extension absent): the caller treats it as unreported, never as the static heap size.
    pub(super) fn vram_budget_adapter(adapter: &wgpu::Adapter) -> Option<u64> {
        VramQuerier::of(adapter)?.query().map(|info| info.budget)
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

/// The identity of a GPU for a [`VramWatch`]: the wgpu adapter handle itself. Two identical cards are two adapters,
/// whatever vendor, device id, name or bus they report, and the comparison and the clone (a reference count) allocate
/// nothing, so a cache hit costs no allocation. Build it with [`Self::of`].
#[derive(Clone, PartialEq, Eq)]
pub struct AdapterKey(Repr);

#[derive(Clone, PartialEq, Eq)]
enum Repr {
    Adapter(wgpu::Adapter),
    /// A stand-in for tests, which have no GPU.
    #[cfg(test)]
    Test(u32),
}

impl AdapterKey {
    /// The key of `adapter`.
    pub fn of(adapter: &wgpu::Adapter) -> Self {
        Self(Repr::Adapter(adapter.clone()))
    }
}

/// Names the adapter in the watch's log lines (only logged on a failure or a recovery, so the info strings may be
/// built here).
impl std::fmt::Debug for AdapterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Repr::Adapter(adapter) => {
                let info = adapter.get_info();
                write!(
                    f,
                    "{:?} {:04x}:{:04x} {} (bus {})",
                    info.backend, info.vendor, info.device, info.name, info.device_pci_bus_id
                )
            }
            #[cfg(test)]
            Repr::Test(n) => write!(f, "test adapter {n}"),
        }
    }
}

/// A rate-limited reading of the OS VRAM budget of ONE device for a consumer whose idle quota follows it (a resource
/// pool's `set_budget`).
///
/// **Why:** the budget changes while the process runs (other processes, the OS), so a pool's quota is queried again
/// from time to time, but a driver query per retire is a syscall storm. The reading is cached for `every`; a failed
/// one (the platform reports none) is cached the same way - never remembered as a permanent zero - and logged once
/// (`log::warn!`, prefixed with the watch's `label`); the first reading after a failure is logged once as a recovery.
///
/// **One device:** the cache belongs to the adapter that was asked ([`AdapterKey`]: the adapter handle, so two
/// identical cards are told apart). Asked about another adapter, the watch forgets its reading and queries at once, so
/// a cached figure of one GPU is never answered for another. Alternating two adapters through one watch re-reads every
/// time: give each its own. A cache hit allocates nothing.
///
/// **Reentrancy:** the query runs while the watch's lock is held. It must not call the watch ([`Self::budget`] or
/// [`Self::budget_with`]) - that deadlocks.
///
/// **Where used:** `ofx-runtime`'s GPU renderer (its shared idle allowance) and `ofx-finish`'s retained jobs.
pub struct VramWatch {
    label: &'static str,
    every: std::time::Duration,
    state: std::sync::Mutex<WatchState>,
}

/// The last reading of a [`VramWatch`].
struct WatchState {
    /// The device the reading is of.
    device: Option<AdapterKey>,
    budget: Option<u64>,
    at: Option<std::time::Instant>,
    /// The last reading failed (and said so): a success logs the recovery.
    failing: bool,
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
    /// A watch that queries at most once per `every` (zero: every call; `Duration::MAX`: once per device); `label`
    /// names the consumer in the failure and recovery log lines.
    pub const fn new(label: &'static str, every: std::time::Duration) -> Self {
        Self {
            label,
            every,
            state: std::sync::Mutex::new(WatchState {
                device: None,
                budget: None,
                at: None,
                failing: false,
            }),
        }
    }

    /// The OS VRAM budget of the context's device in bytes, read again when the last reading is older than the
    /// interval or was of another adapter; `None` while the platform reports none.
    pub fn budget(&self, ctx: GpuVramContext<'_>) -> Option<u64> {
        self.budget_with(&AdapterKey::of(ctx.adapter), || {
            vram_budget_from_context(ctx)
        })
    }

    /// [`Self::budget`] for the adapter `device` (computed once by the caller: [`AdapterKey::of`]) with the query
    /// supplied (a test drives the rate limit, the re-keying and the failure path with it). `query` must not call this
    /// watch.
    pub fn budget_with(
        &self,
        device: &AdapterKey,
        query: impl FnOnce() -> Option<u64>,
    ) -> Option<u64> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let same = state.device.as_ref() == Some(device);
        if same && state.at.is_some_and(|at| at.elapsed() < self.every) {
            return state.budget;
        }
        if !same {
            state.device = Some(device.clone());
            state.failing = false;
        }
        state.budget = query();
        match (state.budget, state.failing) {
            (None, false) => {
                log::warn!(
                    "{}: the OS reports no VRAM budget for {device:?}; nothing is kept idle (retried every {} ms)",
                    self.label,
                    self.every.as_millis()
                );
                state.failing = true;
            }
            (Some(bytes), true) => {
                log::info!(
                    "{}: the OS reports a VRAM budget for {device:?} again ({bytes} bytes)",
                    self.label
                );
                state.failing = false;
            }
            _ => {}
        }
        state.at = Some(std::time::Instant::now());
        state.budget
    }
}

#[cfg(test)]
mod watch_tests {
    use super::{AdapterKey, Repr, VramWatch};
    use std::cell::Cell;
    use std::time::Duration;

    fn gpu(n: u32) -> AdapterKey {
        AdapterKey(Repr::Test(n))
    }

    #[test]
    fn a_reading_is_cached_inside_the_interval_and_fresh_outside_it() {
        // `MAX` never expires, `ZERO` always has: no sleeping, no dependence on call latency.
        let kept = VramWatch::new("test", Duration::MAX);
        let calls = Cell::new(0);
        let query = |value: u64| {
            calls.set(calls.get() + 1);
            Some(value)
        };
        assert_eq!(kept.budget_with(&gpu(0), || query(10)), Some(10));
        assert_eq!(kept.budget_with(&gpu(0), || query(20)), Some(10));
        assert_eq!(calls.get(), 1);
        let fresh = VramWatch::new("test", Duration::ZERO);
        assert_eq!(fresh.budget_with(&gpu(0), || query(30)), Some(30));
        assert_eq!(fresh.budget_with(&gpu(0), || query(40)), Some(40));
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn another_adapter_is_never_answered_from_the_cache_of_the_first() {
        let watch = VramWatch::new("test", Duration::MAX);
        assert_eq!(watch.budget_with(&gpu(0), || Some(10)), Some(10));
        assert_eq!(watch.budget_with(&gpu(1), || Some(99)), Some(99));
        assert_eq!(
            watch.budget_with(&gpu(1), || Some(5)),
            Some(99),
            "cached for gpu1"
        );
        assert_eq!(
            watch.budget_with(&gpu(0), || Some(11)),
            Some(11),
            "back to gpu0: asked again"
        );
    }

    #[test]
    fn a_failed_query_is_retried_and_recovers_not_remembered_as_zero() {
        let watch = VramWatch::new("test", Duration::ZERO);
        assert_eq!(watch.budget_with(&gpu(0), || None), None);
        assert!(
            watch.state.lock().expect("lock").failing,
            "the failure was logged once"
        );
        assert_eq!(watch.budget_with(&gpu(0), || None), None);
        assert_eq!(watch.budget_with(&gpu(0), || Some(7)), Some(7));
        assert!(
            !watch.state.lock().expect("lock").failing,
            "the recovery reset the flag"
        );
        // A later failure is announced again.
        assert_eq!(watch.budget_with(&gpu(0), || None), None);
        assert!(watch.state.lock().expect("lock").failing);
    }
}
