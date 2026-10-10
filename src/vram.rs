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
//! - **Windows** — DXGI `IDXGIAdapter3::QueryVideoMemoryInfo`, the adapter matched by LUID
//!   (backend-agnostic — works when wgpu uses Vulkan too). VERIFIED.
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
    use super::{GpuVramContext, VramInfo, phys_id};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3, IDXGIFactory4,
    };
    use windows::core::Interface;

    pub struct VramQuerier {
        adapter: IDXGIAdapter3,
    }

    /// The DXGI adapter with this LUID (`VkPhysicalDeviceIDProperties::deviceLUID` / `DXGI_ADAPTER_DESC1`). The LUID
    /// is unique per adapter in a boot session, so two identical cards are told apart; a name is not.
    fn match_dxgi_adapter(luid: u64) -> Option<IDXGIAdapter3> {
        let factory: IDXGIFactory4 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.ok()?;
        let mut i = 0u32;
        while let Ok(base) = unsafe { factory.EnumAdapters1(i) } {
            i += 1;
            let Ok(desc) = (unsafe { base.GetDesc1() }) else {
                continue;
            };
            let found = (u64::from(desc.AdapterLuid.HighPart as u32) << 32)
                | u64::from(desc.AdapterLuid.LowPart);
            if found == luid {
                return base.cast::<IDXGIAdapter3>().ok();
            }
        }
        None
    }

    /// The LUID of `adapter`, `None` when neither backend reports one (the budget is then unreported, never another
    /// card's).
    fn luid_of(adapter: &wgpu::Adapter) -> Option<u64> {
        match phys_id(adapter) {
            super::PhysId::Luid(luid) => Some(luid),
            super::PhysId::Uuid(_) | super::PhysId::Unknown => None,
        }
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
            let adapter = match_dxgi_adapter(luid_of(ctx.adapter)?)?;
            Some(Self { adapter })
        }

        pub fn query(&self) -> Option<VramInfo> {
            query_local_segment(&self.adapter)
        }
    }

    pub(super) fn vram_budget_adapter(adapter: &wgpu::Adapter) -> Option<u64> {
        let dxgi = match_dxgi_adapter(luid_of(adapter)?)?;
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
            // `vkGetPhysicalDeviceMemoryProperties2` is core in 1.1 and the budget struct needs the extension: without
            // them the chain would be ignored and read back zeros, so say "unreported" here instead.
            let version = unsafe { instance.get_physical_device_properties(raw_phys) }.api_version;
            let extensions =
                unsafe { instance.enumerate_device_extension_properties(raw_phys) }.ok()?;
            let names = extensions
                .iter()
                .filter_map(|e| e.extension_name_as_c_str().ok()?.to_str().ok());
            if !super::budget_chainable(version, names) {
                return None;
            }
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

/// Whether `VkPhysicalDeviceMemoryBudgetPropertiesEXT` may be chained to `vkGetPhysicalDeviceMemoryProperties2` for a
/// device of Vulkan `api_version` that lists `extensions`: the query is core from 1.1 and the budget struct needs
/// `VK_EXT_memory_budget`; without either the chain would be ignored and read back zeros. Pure, so it is tested on
/// every host; the Linux querier asks it once when it is built.
#[cfg_attr(
    not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        test
    )),
    allow(dead_code)
)]
fn budget_chainable<'a>(api_version: u32, mut extensions: impl Iterator<Item = &'a str>) -> bool {
    // VK_MAKE_API_VERSION: variant 31..29, major 28..22, minor 21..12, patch 11..0.
    let (major, minor) = ((api_version >> 22) & 0x7f, (api_version >> 12) & 0x3ff);
    (major > 1 || (major == 1 && minor >= 1))
        && extensions.any(|name| name == "VK_EXT_memory_budget")
}

/// The physical identity of an adapter, from the driver, computed once per [`AdapterKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PhysId {
    /// The adapter's LUID (Windows; `VkPhysicalDeviceIDProperties::deviceLUID` or `DXGI_ADAPTER_DESC1`), unique per
    /// adapter in a boot session.
    Luid(u64),
    /// `VkPhysicalDeviceIDProperties::deviceUUID` (Vulkan without a valid LUID, i.e. Linux).
    Uuid([u8; 16]),
    /// The backend reports none (Metal, GL).
    Unknown,
}

/// The identity of a GPU for a [`VramWatch`]: the wgpu adapter handle AND the driver's physical identity.
///
/// **Contract:** wgpu compares adapters by an id that is unique inside one `wgpu::Instance`, so adapters of two
/// instances can share an id; the physical identity (LUID or UUID) tells those apart (two instances of the same
/// physical card compare equal, which is right: it is one budget). Two identical cards differ in the physical
/// identity whatever name, vendor or device id they report. Where the backend reports none ([`PhysId::Unknown`])
/// only the handle compares, so keys of different instances on such a backend may collide: use one instance per
/// watch there. Build it once with [`Self::of`]; comparing and cloning allocate nothing, so a cache hit costs no
/// allocation.
#[derive(Clone, PartialEq, Eq)]
pub struct AdapterKey(Repr);

#[derive(Clone, PartialEq, Eq)]
enum Repr {
    Adapter(wgpu::Adapter, PhysId),
    /// A stand-in for tests, which have no GPU.
    #[cfg(test)]
    Test(u32),
}

impl AdapterKey {
    /// The key of `adapter` (queries the driver for its physical identity: build it once, not per call).
    pub fn of(adapter: &wgpu::Adapter) -> Self {
        Self(Repr::Adapter(adapter.clone(), phys_id(adapter)))
    }
}

/// The physical identity of `adapter`: the DX12 adapter's LUID, else the Vulkan device's LUID or UUID.
fn phys_id(adapter: &wgpu::Adapter) -> PhysId {
    #[cfg(windows)]
    if let Some(luid) = dx12_luid(adapter) {
        return PhysId::Luid(luid);
    }
    #[cfg(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    if let Some(id) = vulkan_id(adapter) {
        return id;
    }
    let _ = adapter;
    PhysId::Unknown
}

#[cfg(windows)]
fn dx12_luid(adapter: &wgpu::Adapter) -> Option<u64> {
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
    let desc = unsafe { hal.raw_adapter().GetDesc1() }.ok()?;
    Some((u64::from(desc.AdapterLuid.HighPart as u32) << 32) | u64::from(desc.AdapterLuid.LowPart))
}

#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
fn vulkan_id(adapter: &wgpu::Adapter) -> Option<PhysId> {
    use ash::vk;
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let phys = hal.raw_physical_device();
    let instance = hal.shared_instance().raw_instance();
    // `vkGetPhysicalDeviceProperties2` is core in Vulkan 1.1.
    let version = unsafe { instance.get_physical_device_properties(phys) }.api_version;
    if vk::api_version_major(version) == 1 && vk::api_version_minor(version) < 1 {
        return None;
    }
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    unsafe { instance.get_physical_device_properties2(phys, &mut props) };
    Some(if id.device_luid_valid == vk::TRUE {
        PhysId::Luid(u64::from_le_bytes(id.device_luid))
    } else {
        PhysId::Uuid(id.device_uuid)
    })
}

/// Names the adapter in the watch's log lines (only logged on a failure or a recovery, so the info strings may be
/// built here).
impl std::fmt::Debug for AdapterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Repr::Adapter(adapter, phys) => {
                let info = adapter.get_info();
                write!(
                    f,
                    "{:?} {:04x}:{:04x} {} ({phys:?})",
                    info.backend, info.vendor, info.device, info.name
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
/// **Where used:** consumers whose idle pool quota follows the OS budget, e.g. `ofx-runtime`'s GPU renderer.
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

    /// The Linux budget query chains its struct only on Vulkan >= 1.1 with the extension listed. RED: the version
    /// test dropped (1.0 passes) or the extension name not required.
    #[test]
    fn the_budget_struct_is_chained_only_where_it_is_supported() {
        let v = |major: u32, minor: u32| (major << 22) | (minor << 12);
        let with = ["VK_KHR_swapchain", "VK_EXT_memory_budget"];
        assert!(super::budget_chainable(v(1, 1), with.into_iter()));
        assert!(super::budget_chainable(v(1, 3), with.into_iter()));
        assert!(super::budget_chainable(v(2, 0), with.into_iter()));
        assert!(!super::budget_chainable(v(1, 0), with.into_iter()), "1.0");
        assert!(
            !super::budget_chainable(v(1, 3), ["VK_KHR_swapchain"].into_iter()),
            "no extension"
        );
        assert!(!super::budget_chainable(v(1, 3), [].into_iter()));
    }

    /// Adapters of one instance enumerated twice and of two instances are the same card: their physical identity is
    /// equal and each reports the budget (keys of distinct handles may differ, which only costs the watch a re-read -
    /// documented on [`AdapterKey`]); an identity-less backend would show as `Unknown`, which this machine's must not.
    #[test]
    #[ignore = "needs a GPU"]
    fn the_same_card_has_the_same_identity_across_enumerations_and_instances() {
        let adapters = || {
            let instance = wgpu::Instance::new(crate::shared_instance_descriptor());
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        };
        let phys = |adapter: &wgpu::Adapter| match AdapterKey::of(adapter).0 {
            Repr::Adapter(_, phys) => phys,
            Repr::Test(_) => unreachable!("a real adapter"),
        };
        let (first, second) = (adapters(), adapters());
        let real = |adapter: &&wgpu::Adapter| {
            adapter.get_info().device_type != wgpu::DeviceType::Cpu
                && phys(adapter) != super::PhysId::Unknown
        };
        let a = first.iter().find(real).expect("a hardware adapter");
        let b = second
            .iter()
            .find(|candidate| phys(candidate) == phys(a))
            .expect("the same card in the second instance");
        assert_eq!(phys(a), phys(b));
        let watch = VramWatch::new("test", Duration::ZERO);
        let device = |adapter: &wgpu::Adapter| {
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device")
        };
        for adapter in [a, b] {
            let (device, _queue) = device(adapter);
            let budget = watch.budget(super::GpuVramContext {
                adapter,
                device: &device,
            });
            assert!(
                budget.is_some_and(|bytes| bytes > 0),
                "{:?}",
                AdapterKey::of(adapter)
            );
        }
    }

    /// A real adapter has a physical identity (LUID on Windows, UUID on Linux), two keys of it are equal, and its
    /// budget is reported through the identity-matched query. RED: matching the DXGI adapter by name, or `Unknown`.
    #[test]
    #[ignore = "needs a GPU"]
    fn a_real_adapter_is_identified_and_reports_its_budget() {
        let gpu = crate::compute_device().expect("a GPU");
        let (a, b) = (AdapterKey::of(&gpu.adapter), AdapterKey::of(&gpu.adapter));
        assert_eq!(a, b);
        let Repr::Adapter(_, phys) = &a.0 else {
            panic!("a real key");
        };
        assert_ne!(*phys, super::PhysId::Unknown, "{a:?}");
        assert!(super::vram_budget_bytes(&gpu.adapter).is_some_and(|bytes| bytes > 0));
    }
}
