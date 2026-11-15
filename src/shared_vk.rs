//! The Vulkan half of the ONE shared device: a logical device that serves wgpu compute, optional
//! hardware video decode, and optional external memory / timeline-semaphore interop (CUDA, OpenGL),
//! plus the raw `ash` handles direct Vulkan consumers need.
//!
//! # Why this exists
//!
//! [`crate::shared_device`] used to be a plain `request_device`, so the process had one wgpu device
//! while a video decoder (ffmpeg-rs `av-hwaccel-vulkan`) or CUDA/OptiX bridge (`nv-wgpu`) either had
//! to open a SECOND `VkDevice` or lacked external-memory/semaphore extensions on wgpu's device.
//! Two devices without shared external handles cannot share an image: frames had to travel
//! GPU -> RAM -> GPU on the playback / path-tracing path every frame.
//!
//! wgpu never asks for the video extensions, a decode queue, or external-memory/semaphore Win32/FD
//! sharing extensions, and it does not have to: `wgpu_hal::vulkan::Adapter::open_with_callback`
//! hands us the device-create info before `vkCreateDevice`, so we add whichever video and external
//! interop extensions the physical device actually supports, create the device ONCE, and let wgpu
//! adopt it (`Adapter::create_device_from_hal`). Video decode and external CUDA/GL interop are
//! negotiated as independent optional capabilities so headless compute GPUs without video decode
//! still get zero-copy external interop, and non-interop GPUs never fail device creation.
//!
//! Measured on this exact path before it was written (spike; RTX 3080 Ti, driver 616.64,
//! wgpu 30.0.1): an NV12 image created outside wgpu, filled through the DECODE queue, wrapped with
//! `texture_from_raw` + `create_texture_from_hal` and read by a compute pass - 0 wrong of 4096
//! pixels, with no host round trip.
//!
//! # What is NOT here
//!
//! Only Vulkan uses this module. On a Metal or DX12 adapter `open_shared` returns `None` and
//! the caller negotiates the ordinary native `request_device` path.

use std::ffi::CStr;

use ash::vk;
use wgpu::hal::api::Vulkan;

/// Extensions the video decoder needs on top of whatever wgpu requires.
///
/// Both codec extensions are listed because the device is created once, at startup, before anyone
/// knows which clip will be opened. A codec whose extension the adapter lacks is simply dropped
/// from the request (see [`present_video_ext`]) rather than failing the whole negotiation.
const VIDEO_EXT: [&CStr; 6] = [
    c"VK_KHR_video_queue",
    c"VK_KHR_video_decode_queue",
    c"VK_KHR_video_decode_h265",
    c"VK_KHR_video_decode_h264",
    c"VK_KHR_video_decode_av1",
    c"VK_KHR_video_decode_vp9",
];

/// `VkPhysicalDeviceVideoDecodeVP9FeaturesKHR` (`VK_KHR_video_decode_vp9`, Vulkan 1.4.316;
/// `sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR = 1000514000`).
#[repr(C)]
struct PhysicalDeviceVideoDecodeVp9FeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut core::ffi::c_void,
    video_decode_vp9: vk::Bool32,
}

// SAFETY: plain C feature struct whose `p_next` is wired inside the single-threaded `open_with_callback` closure.
unsafe impl Send for PhysicalDeviceVideoDecodeVp9FeaturesKHR {}
unsafe impl Sync for PhysicalDeviceVideoDecodeVp9FeaturesKHR {}

/// Platform-specific Vulkan external memory and external semaphore device extensions required for
/// zero-copy CUDA/OptiX (`nv-wgpu`) and OpenGL (`GlInterop`) sharing:
/// - Windows: `VK_KHR_external_memory_win32` + `VK_KHR_external_semaphore_win32`
/// - Unix: `VK_KHR_external_memory_fd` + `VK_KHR_external_semaphore_fd`
///
/// On Vulkan < 1.1 drivers these additionally require promoted base extensions
/// (`VK_KHR_get_memory_requirements2`, `VK_KHR_dedicated_allocation`, `VK_KHR_external_memory`,
/// `VK_KHR_external_semaphore`), and timeline semaphore synchronization on Vulkan < 1.2 requires
/// `VK_KHR_timeline_semaphore`. Negotiated without failing device creation when absent via
/// [`negotiate_interop_extensions`].
#[cfg(windows)]
const EXT_MEMORY_PLATFORM: &CStr = c"VK_KHR_external_memory_win32";
#[cfg(unix)]
const EXT_MEMORY_PLATFORM: &CStr = c"VK_KHR_external_memory_fd";

#[cfg(windows)]
const EXT_SEMAPHORE_PLATFORM: &CStr = c"VK_KHR_external_semaphore_win32";
#[cfg(unix)]
const EXT_SEMAPHORE_PLATFORM: &CStr = c"VK_KHR_external_semaphore_fd";

/// Vulkan < 1.1 device extensions required before enabling external memory + dedicated allocations.
const EXT_MEMORY_VK10_PREREQS: [&CStr; 3] = [
    c"VK_KHR_get_memory_requirements2",
    c"VK_KHR_dedicated_allocation",
    c"VK_KHR_external_memory",
];

/// Vulkan < 1.1 device extension required before enabling external semaphores.
const EXT_SEMAPHORE_VK10_PREREQ: &CStr = c"VK_KHR_external_semaphore";

/// Vulkan < 1.2 device extension required for timeline semaphores when not yet core.
const EXT_TIMELINE_SEMAPHORE: &CStr = c"VK_KHR_timeline_semaphore";

/// Instance extension needed on Vulkan 1.0 instances to read `VkPhysicalDeviceIDProperties::deviceUUID`.
const INST_GET_PROPS2: &CStr = c"VK_KHR_get_physical_device_properties2";

/// The two extensions without which there is no decoding at all. If either is missing there is no
/// point adding the rest.
const VIDEO_CORE: [&CStr; 2] = [c"VK_KHR_video_queue", c"VK_KHR_video_decode_queue"];

/// One priority, `'static` so the pointer inside `VkDeviceQueueCreateInfo` is still valid when
/// `vkCreateDevice` reads it (the callback returns before the call happens).
static PRIORITY: [f32; 1] = [1.0];

/// Two priorities, for the case where a compute queue can only be a SECOND queue in wgpu's own
/// family. The count in `VkDeviceQueueCreateInfo` is the length of this slice, so raising the
/// count means replacing the slice - bumping the number alone would have the driver read a float
/// past the end of a one-element array.
static PRIORITY_PAIR: [f32; 2] = [1.0, 1.0];
static PRIORITY_TRIPLE: [f32; 3] = [1.0, 1.0, 1.0];

/// Process-local registry of dedicated Vulkan interop barrier queues `(family, index)`
/// keyed by raw `VkDevice` handle, so `device_vulkan_interop_queue(&wgpu::Device)` can
/// query it without changing `wgpu::Device` ownership.
static INTEROP_QUEUES: std::sync::Mutex<Vec<(u64, (u32, u32))>> = std::sync::Mutex::new(Vec::new());

/// The queue family wgpu renders on.
///
/// wgpu-hal 30 hard-codes it (`vulkan/adapter.rs`, `let family_index = 0; //TODO`) and takes
/// index 0 of it. A constant rather than a guess, and CHECKED after the device is open against
/// `Device::queue_family_index()`: if wgpu ever stops doing this, sharing must stop rather than
/// quietly hand a consumer the renderer's own queue.
pub(crate) const WGPU_FAMILY: u32 = 0;

/// The raw Vulkan objects behind the shared device, for consumers that speak Vulkan directly.
///
/// Every handle here belongs to the SAME logical device wgpu is using. None of them is owned by
/// this struct: wgpu owns the device and destroys it when the process ends, which is also why a
/// consumer must not call `vkDestroyDevice` on it.
///
/// `ash::Entry`, `ash::Instance` and `ash::Device` are handle wrappers (function tables), so
/// cloning them is cheap and does not duplicate the underlying object.
#[derive(Clone)]
pub struct VulkanShared {
    /// The loaded Vulkan library the instance came from.
    pub entry: ash::Entry,
    /// The instance wgpu negotiated the adapter on.
    pub instance: ash::Instance,
    /// The physical device behind the shared adapter.
    pub physical_device: vk::PhysicalDevice,
    /// The logical device wgpu is using. Do NOT destroy it.
    pub device: ash::Device,
    /// The `(family, index)` pair wgpu's OWN queue is, read from wgpu rather than inferred.
    ///
    /// A consumer must never submit to it. `vkQueueSubmit` is externally synchronised, and this
    /// queue already has a submitter - the renderer - on another thread.
    pub main_queue: (u32, u32),
    /// A `(family, index)` that can `vkCmdDecodeVideoKHR`, when the device was created with one.
    ///
    /// `None` means the adapter has no video-decode queue of its own: a decoder must then open
    /// its own device, and every frame it produces has to be copied through host memory.
    ///
    /// The index is part of the answer, not an assumption. A consumer that guessed `0` would
    /// submit into whichever queue happened to be first in the family - possibly the renderer's.
    pub decode_queue: Option<(u32, u32)>,
    /// A `(family, index)` a consumer may run COMPUTE on, distinct from [`Self::main_queue`].
    ///
    /// This is what lets a decoder read a decoded picture in place instead of copying it: the
    /// video-decode family cannot dispatch compute, and the renderer's queue is not ours to
    /// submit to. `None` means no such queue could be reserved, and a consumer must say so rather
    /// than fall back to somebody else's queue.
    pub compute_queue: Option<(u32, u32)>,
    /// A dedicated `(family, index)` in [`Self::main_queue`]'s family for external-interop
    /// (`nv-wgpu`) ownership/layout barriers, distinct from `main_queue` so raw `vkQueueSubmit`
    /// never races `wgpu::Queue::submit` or `SurfaceTexture::present`.
    pub interop_queue: Option<(u32, u32)>,
    /// The video extensions actually enabled on the device, in request order.
    pub video_extensions: Vec<&'static CStr>,
    /// Every device extension enabled at `vkCreateDevice`, not only the video ones.
    ///
    /// A consumer that speaks Vulkan needs this whole list, because Vulkan cannot be asked which
    /// extensions a device was created with, and calling into one that was not enabled is
    /// undefined behaviour. So the creator of the device has to say.
    pub device_extensions: Vec<std::ffi::CString>,
    /// The API version the INSTANCE was created with.
    ///
    /// This is the LOADER's answer (`vkEnumerateInstanceVersion`), and on its own it says nothing
    /// about what the device can do - see [`Self::device_api_version`], which is the one a
    /// consumer must gate on.
    pub instance_api_version: u32,
    /// The PHYSICAL DEVICE's own `apiVersion`.
    ///
    /// This is what decides whether a device entry point exists. `vkCmdPipelineBarrier2` is core
    /// in 1.3, and a 1.3 loader in front of a 1.2 driver reports 1.3 for the instance while the
    /// device has no such function - ash fills an unloadable slot with a stub that PANICS at the
    /// first call, inside whichever library called it. Published separately rather than folded
    /// into one number so a consumer cannot accidentally gate on the wrong one; the answer to
    /// "what may I actually call" is [`Self::usable_api_version`].
    pub device_api_version: u32,
    /// Whether the three features every compute/decode consumer here needs were enabled.
    ///
    /// The device is built with every stable feature the adapter reports, so these are normally
    /// all true; they are recorded rather than assumed because "normally" is not a contract and
    /// `vkGetPhysicalDeviceFeatures` answers what the hardware CAN do, not what was enabled.
    pub timeline_semaphore: bool,
    /// See [`Self::timeline_semaphore`].
    pub synchronization2: bool,
    /// See [`Self::timeline_semaphore`].
    pub sampler_ycbcr_conversion: bool,
    /// Whether full external memory + external timeline-semaphore interop (`nv-wgpu` Vulkan<->CUDA
    /// contract) is enabled on this logical device.
    pub external_interop: bool,
}

impl core::fmt::Debug for VulkanShared {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VulkanShared")
            .field("main_queue", &self.main_queue)
            .field("decode_queue", &self.decode_queue)
            .field("compute_queue", &self.compute_queue)
            .field("video_extensions", &self.video_extensions)
            .field("instance_api_version", &self.instance_api_version)
            .field("device_api_version", &self.device_api_version)
            .field("external_interop", &self.external_interop)
            .field("device_extensions", &self.device_extensions.len())
            .finish()
    }
}

/// The video extensions this adapter really has, in [`VIDEO_EXT`] order.
///
/// Asking for an extension the device does not expose makes `vkCreateDevice` fail outright, so the
/// list is filtered before it is requested.
fn present_video_ext(adapter: &wgpu::hal::vulkan::Adapter) -> Vec<&'static CStr> {
    let caps = adapter.physical_device_capabilities();
    VIDEO_EXT
        .iter()
        .copied()
        .filter(|e| caps.supports_extension(e))
        .collect()
}

/// Outcome of negotiating optional Vulkan external-memory and external-semaphore extensions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InteropNegotiation {
    /// Device extensions verified present on the adapter (with version prerequisites satisfied) to
    /// append to `vkCreateDevice`.
    extensions: Vec<&'static CStr>,
    /// True iff ALL platform external-memory, external-semaphore, timeline-semaphore, and physical
    /// device UUID capabilities required by zero-copy Vulkan<->CUDA interop are active.
    external_interop: bool,
}

/// Pure, hardware-independent negotiation of Vulkan external-memory and external-semaphore device
/// extensions across Vulkan 1.0, 1.1, 1.2, and 1.3+ drivers.
///
/// Never requests an extension the physical device lacks or whose `< 1.1` base prerequisite is
/// missing, so `vkCreateDevice` never fails on non-NVIDIA or software adapters. Partial support
/// (for example external semaphore without external memory) still enables the supported half while
/// reporting `external_interop = false`.
fn negotiate_interop_extensions(
    instance_api_version: u32,
    has_inst_props2: bool,
    device_api_version: u32,
    supports_ext: impl Fn(&CStr) -> bool,
) -> InteropNegotiation {
    let mut extensions = Vec::with_capacity(7);
    let push_unique = |list: &mut Vec<&'static CStr>, ext: &'static CStr| {
        if !list.contains(&ext) {
            list.push(ext);
        }
    };

    #[cfg(any(windows, unix))]
    let (mem_ok, sem_ok) = {
        let mem_prereqs_ok = device_api_version >= vk::API_VERSION_1_1
            || EXT_MEMORY_VK10_PREREQS.iter().all(|e| supports_ext(e));
        let mem_ok = mem_prereqs_ok && supports_ext(EXT_MEMORY_PLATFORM);
        if mem_ok {
            if device_api_version < vk::API_VERSION_1_1 {
                for &req in &EXT_MEMORY_VK10_PREREQS {
                    push_unique(&mut extensions, req);
                }
            }
            push_unique(&mut extensions, EXT_MEMORY_PLATFORM);
        }

        let sem_prereqs_ok =
            device_api_version >= vk::API_VERSION_1_1 || supports_ext(EXT_SEMAPHORE_VK10_PREREQ);
        let sem_ok = sem_prereqs_ok && supports_ext(EXT_SEMAPHORE_PLATFORM);
        if sem_ok {
            if device_api_version < vk::API_VERSION_1_1 {
                push_unique(&mut extensions, EXT_SEMAPHORE_VK10_PREREQ);
            }
            push_unique(&mut extensions, EXT_SEMAPHORE_PLATFORM);
        }
        (mem_ok, sem_ok)
    };

    #[cfg(not(any(windows, unix)))]
    let (mem_ok, sem_ok) = (false, false);

    let timeline_ok = if device_api_version >= vk::API_VERSION_1_2 {
        true
    } else if supports_ext(EXT_TIMELINE_SEMAPHORE) {
        if mem_ok || sem_ok {
            push_unique(&mut extensions, EXT_TIMELINE_SEMAPHORE);
        }
        true
    } else {
        false
    };

    let uuid_ok = instance_api_version >= vk::API_VERSION_1_1 || has_inst_props2;
    InteropNegotiation {
        extensions,
        external_interop: mem_ok && sem_ok && timeline_ok && uuid_ok,
    }
}

/// Negotiates external-interop extensions against a live `wgpu_hal::vulkan::Adapter`.
fn present_interop_ext(adapter: &wgpu::hal::vulkan::Adapter) -> InteropNegotiation {
    let caps = adapter.physical_device_capabilities();
    let shared_inst = adapter.shared_instance();
    let instance_api_version = shared_inst.instance_api_version();
    let has_inst_props2 = shared_inst.extensions().contains(&INST_GET_PROPS2);
    let device_api_version = caps.properties().api_version;
    negotiate_interop_extensions(
        instance_api_version,
        has_inst_props2,
        device_api_version,
        |ext| caps.supports_extension(ext),
    )
}

/// Returns the dedicated Vulkan `(family, index)` reserved for external-interop barriers on
/// `device`, if this device was opened with one via [`open_shared`].
#[must_use]
pub(crate) fn vulkan_device_interop_queue(device: &wgpu::Device) -> Option<(u32, u32)> {
    use ash::vk::Handle as _;
    // SAFETY: Read-only inspection of wgpu's Vulkan HAL device handle.
    let hal_dev = unsafe { device.as_hal::<Vulkan>() }?;
    let raw = hal_dev.raw_device().handle().as_raw();
    INTEROP_QUEUES
        .lock()
        .ok()?
        .iter()
        .find(|(dev, _)| *dev == raw)
        .map(|(_, q)| *q)
}

/// Checks whether an already-created Vulkan `wgpu::Device` has all extensions and API capabilities
/// required for zero-copy external memory + timeline-semaphore interop (`nv-wgpu` contract).
#[must_use]
pub(crate) fn vulkan_device_has_external_interop(device: &wgpu::Device) -> bool {
    // SAFETY: Read-only inspection of wgpu's Vulkan HAL device and physical-device properties.
    let Some(hal_dev) = (unsafe { device.as_hal::<Vulkan>() }) else {
        return false;
    };
    let enabled = hal_dev.enabled_device_extensions();
    let shared_inst = hal_dev.shared_instance();
    let instance_api_version = shared_inst.instance_api_version();
    let has_inst_props2 = shared_inst.extensions().contains(&INST_GET_PROPS2);
    if instance_api_version < vk::API_VERSION_1_1 && !has_inst_props2 {
        return false;
    }

    let phys = hal_dev.raw_physical_device();
    // SAFETY: `phys` belongs to `shared_inst.raw_instance()`.
    let device_api_version = unsafe {
        shared_inst
            .raw_instance()
            .get_physical_device_properties(phys)
    }
    .api_version;

    #[cfg(any(windows, unix))]
    {
        if !enabled.contains(&EXT_MEMORY_PLATFORM) || !enabled.contains(&EXT_SEMAPHORE_PLATFORM) {
            return false;
        }
        if device_api_version < vk::API_VERSION_1_1
            && (!EXT_MEMORY_VK10_PREREQS.iter().all(|e| enabled.contains(e))
                || !enabled.contains(&EXT_SEMAPHORE_VK10_PREREQ))
        {
            return false;
        }
        if device_api_version < vk::API_VERSION_1_2 && !enabled.contains(&EXT_TIMELINE_SEMAPHORE) {
            return false;
        }
        true
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (enabled, device_api_version);
        false
    }
}

/// Can `synchronization2` be enabled on this adapter at all?
///
/// True when the device is 1.3 (where it is a core feature) or exposes `VK_KHR_synchronization2`.
/// False means the chained feature struct would be ignored, and a consumer recording `2`-form
/// barriers would be relying on a function that does not exist.
fn sync2_possible(adapter: &wgpu::hal::vulkan::Adapter) -> bool {
    let caps = adapter.physical_device_capabilities();
    caps.properties().api_version >= vk::API_VERSION_1_3
        || caps.supports_extension(c"VK_KHR_synchronization2")
}

/// Does enabling it also require asking for the EXTENSION?
///
/// Only below 1.3: there the feature exists solely as `VK_KHR_synchronization2`, and a feature
/// struct without its extension enables nothing.
fn sync2_needs_extension(adapter: &wgpu::hal::vulkan::Adapter) -> bool {
    let caps = adapter.physical_device_capabilities();
    caps.properties().api_version < vk::API_VERSION_1_3
        && caps.supports_extension(c"VK_KHR_synchronization2")
}

/// Every queue family's flags and queue count, in index order.
fn families(adapter: &wgpu::hal::vulkan::Adapter) -> Vec<vk::QueueFamilyProperties> {
    // SAFETY: the physical device and instance belong to this adapter and outlive the call.
    unsafe {
        adapter
            .shared_instance()
            .raw_instance()
            .get_physical_device_queue_family_properties(adapter.raw_physical_device())
    }
}

/// A family a CONSUMER may dispatch compute on, given the families already spoken for.
///
/// Preference order, and the reason for it:
/// 1. a family with COMPUTE and no GRAPHICS that is neither wgpu's nor the decoder's - a queue
///    nobody else submits to, which is the only arrangement with no contention at all;
/// 2. any other COMPUTE family that is not wgpu's and not the decode family;
/// 3. nothing - and the caller says so rather than handing over a queue somebody else owns.
///
/// The decode family is excluded deliberately even when it reports COMPUTE: two roles on one
/// queue are two unsynchronised submitters, and a decoder that also ran its passes there would
/// serialise its own decoding behind them.
fn compute_family(
    families: &[vk::QueueFamilyProperties],
    main: u32,
    decode: Option<u32>,
) -> Option<u32> {
    let usable = |i: usize, f: &vk::QueueFamilyProperties| {
        let idx = u32::try_from(i).ok()?;
        (f.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && idx != main
            && Some(idx) != decode
            && f.queue_count > 0)
            .then_some(idx)
    };
    families
        .iter()
        .enumerate()
        .find_map(|(i, f)| {
            usable(i, f).filter(|_| !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        })
        .or_else(|| families.iter().enumerate().find_map(|(i, f)| usable(i, f)))
}

/// The first queue family that can decode video, if any.
fn decode_family(adapter: &wgpu::hal::vulkan::Adapter) -> Option<u32> {
    // SAFETY: the physical device and instance belong to this adapter and outlive the call.
    let families = unsafe {
        adapter
            .shared_instance()
            .raw_instance()
            .get_physical_device_queue_family_properties(adapter.raw_physical_device())
    };
    families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))
        .map(|i| i as u32)
}

/// Create the one logical Vulkan device that serves wgpu alongside optional hardware video decode
/// and optional external-memory / timeline-semaphore interop.
///
/// Video decode and external CUDA/GL interop are negotiated independently:
/// - Headless / compute Vulkan GPUs that lack `VK_KHR_video_queue` or a decode queue family still
///   open with external-memory and external-semaphore extensions enabled (`decode_queue: None`,
///   `external_interop: true`).
/// - Adapters that have video decode without external interop still open with the video-decode
///   queue and extensions (`decode_queue: Some(..)`, `external_interop: false`).
/// - Returns `None` when this is not a Vulkan adapter, when neither video decode nor any external
///   interop extension is supported, or when device creation fails so the caller falls back cleanly.
pub(crate) fn open_shared(
    adapter: &wgpu::Adapter,
    features: wgpu::Features,
    limits: &wgpu::Limits,
) -> Option<(wgpu::Device, wgpu::Queue, VulkanShared)> {
    // SAFETY (both blocks): `as_hal` yields the adapter wgpu is already using; we only read from
    // it, and the guard is dropped before the device is adopted. `open_with_callback`'s contract
    // is that the callback may add, never remove - which is what the closure below does.
    //
    // The full enabled-extension list is recorded from inside the callback, because Vulkan cannot
    // be asked afterwards which extensions a device was created with, and a consumer that calls
    // into one that was not enabled is in undefined behaviour.
    let all_extensions =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::<std::ffi::CString>::new()));
    let extensions_writer = std::sync::Arc::clone(&all_extensions);

    // `synchronization2` is NOT enabled by wgpu (wgpu-hal 30 never builds a
    // `PhysicalDeviceSynchronization2Features`), and a video decoder on this device needs it:
    // every barrier ffmpeg-rs records is the `2` form. Enabled here, in the one place that builds
    // this device, instead of left for each consumer to discover as undefined behaviour.
    //
    // A local, borrowed by the callback - NOT leaked. `vkCreateDevice` reads the chain after the
    // callback returns, so the struct must outlive the callback, and it does: wgpu-hal's signature
    // is `CreateDeviceCallback<'this>` with `'this: 'pnext`, so anything living for this function
    // is long-lived enough. It used to be `Box::leak`ed, which leaked once per call for nothing.
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let sync2_ref = &mut sync2;
    let mut vp9_feat = PhysicalDeviceVideoDecodeVp9FeaturesKHR {
        s_type: vk::StructureType::from_raw(1_000_514_000),
        p_next: core::ptr::null_mut(),
        video_decode_vp9: vk::TRUE,
    };
    let vp9_feat_ref = &mut vp9_feat;

    let (video_ext, decode, compute, interop_queue, sync2_on, external_interop, open) = unsafe {
        let hal = adapter.as_hal::<Vulkan>()?;
        let raw_video_ext = present_video_ext(&hal);
        let has_video_core = VIDEO_CORE.iter().all(|e| raw_video_ext.contains(e));
        let decode = if !has_video_core {
            log::info!("gpu-info: adapter has no Vulkan video extensions; opening without shared decode queue");
            None
        } else {
            match decode_family(&hal) {
                None => {
                    log::info!(
                        "gpu-info: adapter has no Vulkan video-decode queue family; opening without shared decode queue"
                    );
                    None
                }
                // A decode queue that IS the renderer's queue is not a shared device, it is two
                // unsynchronised submitters wearing one handle. Refuse sharing that queue loudly:
                // `vkQueueSubmit` is externally synchronised and nothing downstream could detect this.
                Some(family) if family == WGPU_FAMILY => {
                    log::warn!(
                        "gpu-info: the only video-decode family ({family}) is the one wgpu renders on; \
                         disabling shared video decode so a decoder and the renderer never submit to \
                         one queue from two threads"
                    );
                    None
                }
                Some(family) => Some(family),
            }
        };
        let video_ext = if decode.is_some() {
            raw_video_ext
        } else {
            Vec::new()
        };
        let vp9_on = video_ext.contains(&c"VK_KHR_video_decode_vp9");

        let interop = present_interop_ext(&hal);
        if decode.is_none() && interop.extensions.is_empty() {
            log::info!(
                "gpu-info: Vulkan adapter exposes neither video decode nor external interop \
                 extensions; using standard wgpu device creation"
            );
            return None;
        }

        // wgpu's family and queue index, read from wgpu rather than inferred from the callback's
        // first create-info: `queue_family_index()`/`queue_index()` are public on wgpu-hal 30's
        // Vulkan device and are the authoritative pair. wgpu hard-codes family 0 today
        // (`adapter.rs`, `let family_index = 0; //TODO`), which is exactly the kind of fact that
        // must be read rather than assumed.
        let queue_families = families(&hal);
        // A compute queue for whoever reads a decoded picture in place. A family of its own is
        // best; failing that, a SECOND queue in wgpu's family, which is ours because wgpu keeps
        // index 0; failing that, none - and a consumer is told so rather than handed the
        // renderer's queue.
        let compute = match compute_family(&queue_families, WGPU_FAMILY, decode) {
            Some(f) => Some((f, 0)),
            None => queue_families
                .get(WGPU_FAMILY as usize)
                .filter(|f| f.queue_count >= 2)
                .map(|_| (WGPU_FAMILY, 1)),
        };
        if decode.is_some() && compute.is_none() {
            log::warn!(
                "gpu-info: no compute queue could be reserved beside wgpu's; a consumer that \
                 wanted to read a decoded picture in place will have to copy it instead"
            );
        }
        // Whether the feature can be enabled at all, and whether enabling it also means asking
        // for the extension. Decided HERE, where the adapter is in hand, because the callback
        // below cannot query it.
        let sync2_on = sync2_possible(&hal);
        let sync2_ext = sync2_needs_extension(&hal);
        if decode.is_some() && !sync2_on {
            log::warn!(
                "gpu-info: this adapter cannot enable synchronization2; a video decoder sharing \
                 this device would record barriers it does not support"
            );
        }
        let mut extra = video_ext.clone();
        extra.extend(interop.extensions);
        if sync2_ext {
            extra.push(c"VK_KHR_synchronization2");
        }
        // Where a compute queue has to be a SECOND queue in wgpu's own family, the count is
        // raised by replacing that entry's priority slice - one entry per family is a hard rule
        // (VUID-VkDeviceCreateInfo-queueFamilyIndex-02802), so a second create-info for the same
        // family would be invalid usage rather than a second queue.
        // Reserve a dedicated queue in `WGPU_FAMILY` for `nv-wgpu` ownership/layout barriers
        // whenever external interop is enabled and the family has a spare queue. Keeping it in
        // `WGPU_FAMILY` lets `VK_QUEUE_FAMILY_EXTERNAL <-> WGPU_FAMILY` transitions stay valid for
        // `wgpu` on `(WGPU_FAMILY, 0)` without ever calling raw `vkQueueSubmit` on `wgpu`'s queue.
        let main_family_queues = queue_families
            .get(WGPU_FAMILY as usize)
            .map_or(0, |f| f.queue_count);
        let interop_queue = if interop.external_interop {
            if compute == Some((WGPU_FAMILY, 1)) {
                (main_family_queues >= 3).then_some((WGPU_FAMILY, 2))
            } else {
                (main_family_queues >= 2).then_some((WGPU_FAMILY, 1))
            }
        } else {
            None
        };
        let main_queue_count: usize = if interop_queue == Some((WGPU_FAMILY, 2)) {
            3
        } else if compute == Some((WGPU_FAMILY, 1)) || interop_queue == Some((WGPU_FAMILY, 1)) {
            2
        } else {
            1
        };
        let open = hal
            .open_with_callback(
                features,
                limits,
                &wgpu::MemoryHints::Performance,
                Some(Box::new(move |args| {
                    // Deduplicate against extensions wgpu-hal already requested (for example
                    // `VK_KHR_timeline_semaphore` on Vulkan 1.1 drivers) so `VkDeviceCreateInfo`
                    // never receives duplicate extension names.
                    for &ext in &extra {
                        if !args.extensions.contains(&ext) {
                            args.extensions.push(ext);
                        }
                    }
                    // wgpu asks for its own family only; the decoder needs a queue of its own,
                    // and so does anything that reads a decoded picture with a compute shader.
                    let mut want = |family: u32| {
                        if !args
                            .queue_create_infos
                            .iter()
                            .any(|q| q.queue_family_index == family)
                        {
                            args.queue_create_infos.push(
                                vk::DeviceQueueCreateInfo::default()
                                    .queue_family_index(family)
                                    .queue_priorities(&PRIORITY),
                            );
                        }
                    };
                    if let Some(decode_family) = decode {
                        want(decode_family);
                    }
                    if let Some((family, _)) = compute.filter(|(f, _)| *f != WGPU_FAMILY) {
                        want(family);
                    }
                    if main_queue_count > 1 {
                        // wgpu keeps index 0 of this entry (it passes its own pre-callback copy
                        // of the family index and the literal 0 to `device_from_raw`), so indices
                        // 1..main_queue_count are ours.
                        if let Some(first) = args.queue_create_infos.first_mut() {
                            let priorities: &'static [f32] = if main_queue_count >= 3 {
                                &PRIORITY_TRIPLE
                            } else {
                                &PRIORITY_PAIR
                            };
                            *first = first.queue_priorities(priorities);
                        }
                    }
                    // Chain `synchronization2` on, but only where the device can really have it:
                    // a feature struct asking for an unsupported feature either fails the create
                    // or is ignored, and "ignored" is the dangerous one - it would leave the flag
                    // below claiming a feature nothing enabled.
                    if sync2_on {
                        let taken = core::mem::take(args.create_info);
                        *args.create_info = taken.push_next(sync2_ref);
                    }
                    if vp9_on {
                        vp9_feat_ref.p_next = args.create_info.p_next.cast_mut();
                        args.create_info.p_next =
                            (vp9_feat_ref as *const PhysicalDeviceVideoDecodeVp9FeaturesKHR).cast();
                    }
                    if let Ok(mut all) = extensions_writer.lock() {
                        *all = args
                            .extensions
                            .iter()
                            .map(|e| std::ffi::CString::from(*e))
                            .collect();
                    }
                })),
            )
            .inspect_err(|e| log::warn!("gpu-info: shared Vulkan device creation failed: {e}"))
            .ok()?;
        (
            video_ext,
            decode,
            compute,
            interop_queue,
            sync2_on,
            interop.external_interop,
            open,
        )
    };

    // The assumption above, verified. `queue_family_index()`/`queue_index()` are public on
    // wgpu-hal 30's Vulkan device, so this is wgpu's own answer rather than an inference from the
    // create-infos - and a mismatch means every family decision above was made against the wrong
    // renderer queue.
    let main_queue = (open.device.queue_family_index(), open.device.queue_index());
    if main_queue.0 != WGPU_FAMILY {
        log::warn!(
            "gpu-info: wgpu renders on family {} rather than {WGPU_FAMILY}; not sharing this \
             device, because the decode and compute queues were chosen against the wrong one",
            main_queue.0
        );
        return None;
    }

    // Read the raw handles BEFORE wgpu takes the device over; they stay valid because wgpu keeps
    // the device alive for the life of the process.
    // SAFETY: `as_hal` again only reads; the handles are copies of function tables.
    let raw = unsafe {
        let hal = adapter.as_hal::<Vulkan>()?;
        VulkanShared {
            entry: hal.shared_instance().entry().clone(),
            instance: hal.shared_instance().raw_instance().clone(),
            physical_device: hal.raw_physical_device(),
            device: open.device.raw_device().clone(),
            main_queue,
            decode_queue: decode.map(|d| (d, 0)),
            compute_queue: compute,
            interop_queue: if external_interop { interop_queue } else { None },
            video_extensions: video_ext,
            device_extensions: all_extensions
                .lock()
                .map(|all| all.clone())
                .unwrap_or_default(),
            instance_api_version: hal.shared_instance().instance_api_version(),
            device_api_version: hal.physical_device_capabilities().properties().api_version,
            // wgpu enables timeline semaphores exactly when the adapter supports them
            // (wgpu-hal 30 `vulkan/adapter.rs:381-386`).
            timeline_semaphore: hal
                .physical_device_capabilities()
                .supports_extension(c"VK_KHR_timeline_semaphore")
                || hal.physical_device_capabilities().properties().api_version
                    >= vk::API_VERSION_1_2,
            // Enabled by the callback above, because wgpu does not - but only where enabling it
            // MEANS something. Chaining `PhysicalDeviceSynchronization2Features` onto a device
            // that supports neither the 1.3 core feature nor `VK_KHR_synchronization2` is not an
            // error: a driver may ignore a chain entry it does not recognise, `vkCreateDevice`
            // succeeds, and the flag would then claim a feature nothing enabled. Every `2`-form
            // barrier a consumer records against such a device is undefined behaviour, on exactly
            // the machines this office does not own. So the flag is READ BACK from what the
            // device can do rather than asserted from what was asked for.
            synchronization2: sync2_on,
            // NOT enabled: wgpu builds the feature struct with the enabling line commented out
            // (wgpu-hal 30 `vulkan/adapter.rs:424`), and this crate cannot add a second struct of
            // the same type to the chain. A consumer that needs Ycbcr sampling must open its own
            // device - one that reads the planes separately does not.
            sampler_ycbcr_conversion: false,
            external_interop,
        }
    };

    // SAFETY: `open` was created from this adapter, and the descriptor asks for exactly the
    // features and limits it was opened with. Experimental features are enabled because CubeCL's
    // SPIR-V (and Metal MSL) passthrough needs that opt-in and because `shared_device` asks for
    // `EXPERIMENTAL_RAY_QUERY` where the adapter has it (`ray_query_request`, the only experimental
    // FEATURE requested).
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("gpu-info shared device (compute + video decode + external interop)"),
                required_features: features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
            },
        )
    }
    .inspect_err(|e| log::warn!("gpu-info: wgpu refused the shared Vulkan device: {e}"))
    .ok()?;

    if let Some(iq) = raw.interop_queue {
        use ash::vk::Handle as _;
        if let Ok(mut guard) = INTEROP_QUEUES.lock() {
            let dev_raw = raw.device.handle().as_raw();
            if !guard.iter().any(|(d, _)| *d == dev_raw) {
                guard.push((dev_raw, iq));
            }
        }
    }

    log::info!(
        "gpu-info: shared Vulkan device: wgpu on {:?}, decode on {:?}, compute on {:?}, \
         interop on {:?}, external_interop={}, video extensions {:?}",
        raw.main_queue,
        raw.decode_queue,
        raw.compute_queue,
        raw.interop_queue,
        raw.external_interop,
        raw.video_extensions
    );
    Some((device, queue, raw))
}

impl VulkanShared {
    /// The version whose entry points a consumer may actually call.
    ///
    /// The lower of the instance's and the device's: an entry point needs BOTH the instance to
    /// have been created at that version and the device to implement it. This is the number to
    /// hand to anything that refuses to run below a version - never
    /// [`Self::instance_api_version`] alone, which on a 1.3 loader in front of an older driver
    /// promises functions that are not there.
    #[must_use]
    pub fn usable_api_version(&self) -> u32 {
        self.instance_api_version.min(self.device_api_version)
    }

    /// The decode queue itself, or `None` when this adapter has no decode family.
    ///
    /// Fetched on demand rather than stored: `vkGetDeviceQueue` is a table lookup, and a decoder
    /// that wants the queue usually wants it once, at session setup.
    #[must_use]
    pub fn decode_queue(&self) -> Option<vk::Queue> {
        let (family, index) = self.decode_queue?;
        // SAFETY: the pair was passed to `vkCreateDevice`, so the queue exists.
        Some(unsafe { self.device.get_device_queue(family, index) })
    }

    /// The compute queue a consumer may submit to, or `None` when none was reserved.
    ///
    /// Never wgpu's own: [`Self::main_queue`] has a submitter already, and `vkQueueSubmit` is
    /// externally synchronised.
    #[must_use]
    pub fn compute_queue(&self) -> Option<vk::Queue> {
        let (family, index) = self.compute_queue?;
        // SAFETY: the pair was passed to `vkCreateDevice`, so the queue exists.
        Some(unsafe { self.device.get_device_queue(family, index) })
    }

    /// Whether zero-copy external memory and timeline-semaphore interop (`nv-wgpu` Vulkan<->CUDA
    /// contract) is enabled on this logical Vulkan device.
    #[must_use]
    pub fn has_external_interop(&self) -> bool {
        self.external_interop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a Vulkan adapter that HAS the video extensions and a decode queue, the shared device
    /// must carry them. Written as an implication rather than a flat assertion so it states a
    /// property instead of this machine's hardware - but on a machine where the premise holds
    /// (any modern NVIDIA/AMD/Intel GPU) it cannot pass by being skipped.
    #[test]
    fn a_capable_vulkan_adapter_yields_a_decode_capable_shared_device() {
        let Some(shared) = crate::shared_device() else {
            // No adapter at all: there is nothing this test can assert about a device that does
            // not exist. Loud, so it is never mistaken for a pass on a machine that HAS a GPU.
            eprintln!("no GPU adapter on this machine; nothing to assert");
            return;
        };
        let info = shared.adapter.get_info();
        if info.backend != wgpu::Backend::Vulkan {
            eprintln!(
                "backend is {:?}, not Vulkan; the shared decode path cannot apply",
                info.backend
            );
            assert!(
                shared.vulkan.is_none(),
                "non-Vulkan backend must not claim a Vulkan half"
            );
            return;
        }

        // What the adapter itself says, read independently of the code under test.
        // SAFETY: read-only use of the adapter wgpu already owns.
        let (has_core, has_decode_family) = unsafe {
            let hal = shared.adapter.as_hal::<Vulkan>().expect("vulkan adapter");
            let caps = hal.physical_device_capabilities();
            (
                VIDEO_CORE.iter().all(|e| caps.supports_extension(e)),
                decode_family(&hal).is_some(),
            )
        };

        if has_core && has_decode_family {
            let vk = shared
                .vulkan
                .as_ref()
                .expect("adapter can decode video, so the shared device must carry it");
            let decode = vk
                .decode_queue
                .expect("a shared Vulkan half without a decode queue is useless");
            assert!(
                vk.decode_queue().is_some(),
                "decode queue must be fetchable"
            );
            // The pair wgpu itself reports, not one inferred from the create-infos.
            // SAFETY: read-only use of the adapter wgpu already owns.
            let wgpu_pair = unsafe {
                let hal = shared.device.as_hal::<Vulkan>().expect("vulkan device");
                (hal.queue_family_index(), hal.queue_index())
            };
            assert_eq!(
                vk.main_queue, wgpu_pair,
                "the published main queue must be the one wgpu actually renders on"
            );
            // THE rule this whole struct exists to keep: no consumer queue may be wgpu's own.
            // `vkQueueSubmit` is externally synchronised, so sharing that handle with a decoder
            // on another thread is a data race nothing downstream could detect.
            assert_ne!(
                decode, vk.main_queue,
                "the decode queue must not be the queue wgpu renders on"
            );
            if let Some(compute) = vk.compute_queue {
                assert_ne!(
                    compute, vk.main_queue,
                    "the compute queue must not be the queue wgpu renders on"
                );
                assert_ne!(
                    compute, decode,
                    "the compute queue must not be the decode queue: the decode family cannot \
                     dispatch compute, and two roles on one queue are two submitters"
                );
                assert!(
                    vk.compute_queue().is_some(),
                    "a published compute pair must be fetchable"
                );
            }
            // On any adapter with a compute family of its own - every discrete GPU - one must
            // have been reserved, because without it a consumer cannot read a decoded picture in
            // place and is forced back into copying the frame.
            // SAFETY: read-only use of the adapter wgpu already owns.
            let has_other_compute = unsafe {
                let hal = shared.adapter.as_hal::<Vulkan>().expect("vulkan adapter");
                compute_family(&families(&hal), vk.main_queue.0, Some(decode.0)).is_some()
            };
            if has_other_compute {
                assert!(
                    vk.compute_queue.is_some(),
                    "this adapter has a compute family beside wgpu's and the decoder's, so a \
                     compute queue must have been reserved"
                );
            }
            // Printed because "which queue went where" is the whole contract, and a green test
            // that never shows it leaves the reader to trust the assertions blind.
            eprintln!(
                "[shared-vk] wgpu {:?}, decode {:?}, compute {:?} on {}",
                vk.main_queue, vk.decode_queue, vk.compute_queue, info.name
            );
            assert!(
                VIDEO_CORE.iter().all(|e| vk.video_extensions.contains(e)),
                "the core video extensions must be enabled, got {:?}",
                vk.video_extensions
            );
            // What a decoder needs from this device, stated rather than hoped for: it records
            // every barrier in the `2` form and waits on timeline semaphores, and both are core
            // in 1.3 / 1.2 respectively.
            // The DEVICE's version, not the instance's. A 1.3 loader in front of an older driver
            // reports 1.3 for the instance while the device has no `vkCmdPipelineBarrier2` - and
            // ash fills an unloadable slot with a stub that panics at the first call, inside
            // whichever library made it. `usable_api_version` is the honest answer, so it is what
            // is pinned; the instance figure alone would have passed on such a machine.
            assert!(
                vk.usable_api_version() >= vk::API_VERSION_1_3,
                "a decode consumer needs 1.3 on BOTH sides, got instance {:#x} device {:#x}",
                vk.instance_api_version,
                vk.device_api_version
            );
            assert_eq!(
                vk.usable_api_version(),
                vk.instance_api_version.min(vk.device_api_version),
                "the usable version is the lower of the two, by definition"
            );
            // An IMPLICATION, because the claim is what matters: this flag being true must mean
            // the device can really have the feature. Asserting it flatly would pass on a machine
            // where the chained struct was silently ignored - which is the bug this replaced.
            // SAFETY: read-only use of the adapter wgpu already owns.
            let can_sync2 = unsafe {
                let hal = shared.adapter.as_hal::<Vulkan>().expect("vulkan adapter");
                sync2_possible(&hal)
            };
            assert_eq!(
                vk.synchronization2, can_sync2,
                "synchronization2 must be reported as what the device can actually enable, \
                 not as what was asked for"
            );
            assert!(
                vk.synchronization2,
                "a 1.3 device has synchronization2, and a decoder sharing this device records \
                 every barrier in the `2` form"
            );
            assert!(
                vk.timeline_semaphore,
                "a 1.2+ device has timeline semaphores, and wgpu enables them"
            );
            assert!(
                !vk.device_extensions.is_empty(),
                "the enabled extension list is what a consumer must be told; it cannot be empty"
            );
            // Every interop extension the adapter has (with satisfied prerequisites) is enabled,
            // and `has_external_interop()` matches the actual enabled HAL device state.
            // SAFETY: read-only use of the adapter wgpu already owns.
            let negotiated_interop = unsafe {
                let hal = shared.adapter.as_hal::<Vulkan>().expect("vulkan adapter");
                present_interop_ext(&hal)
            };
            for &ext in &negotiated_interop.extensions {
                assert!(
                    vk.device_extensions.iter().any(|e| e.as_c_str() == ext),
                    "{ext:?} was negotiated on this adapter so it must be enabled on the device"
                );
            }
            assert_eq!(
                vk.has_external_interop(),
                negotiated_interop.external_interop,
                "VulkanShared::has_external_interop() must equal the negotiated capability"
            );
            assert_eq!(
                shared.has_external_interop(),
                vk.has_external_interop(),
                "SharedGpu::has_external_interop() must agree with VulkanShared"
            );
            assert_eq!(
                crate::device_has_external_interop(&shared.device),
                vk.has_external_interop(),
                "device_has_external_interop(&device) must agree with VulkanShared"
            );
        } else {
            // Video decode and external interop are independent capabilities: if video decode is
            // absent, `shared.vulkan` is opened iff external interop extensions were negotiated.
            // SAFETY: read-only use of the adapter wgpu already owns.
            let negotiated_interop = unsafe {
                let hal = shared.adapter.as_hal::<Vulkan>().expect("vulkan adapter");
                present_interop_ext(&hal)
            };
            if negotiated_interop.extensions.is_empty() {
                assert!(
                    shared.vulkan.is_none(),
                    "adapter has neither video decode nor external interop, so no Vulkan half may be claimed"
                );
                assert!(!shared.has_external_interop());
            } else {
                let vk = shared
                    .vulkan
                    .as_ref()
                    .expect("adapter has external interop extensions even without video decode");
                assert!(
                    vk.decode_queue.is_none(),
                    "adapter cannot decode video, so decode_queue must be None"
                );
                assert!(
                    vk.video_extensions.is_empty(),
                    "video extensions must not be enabled without a decode queue"
                );
                assert_eq!(
                    vk.has_external_interop(),
                    negotiated_interop.external_interop
                );
                assert_eq!(shared.has_external_interop(), vk.has_external_interop());
            }
        }
    }

    /// Synthetic Vulkan 1.0 / 1.1 / 1.2 / 1.3 capability matrix proving that external-memory and
    /// external-semaphore negotiation adds promoted `< 1.1` / `< 1.2` base extensions when needed,
    /// never requests unsupported extensions, and sets `external_interop` iff the complete
    /// `nv-wgpu` extension + UUID contract is met.
    #[test]
    fn external_interop_extension_negotiation_matrix() {
        #[cfg(windows)]
        let (mem_plat, sem_plat) = (
            c"VK_KHR_external_memory_win32",
            c"VK_KHR_external_semaphore_win32",
        );
        #[cfg(unix)]
        let (mem_plat, sem_plat) = (
            c"VK_KHR_external_memory_fd",
            c"VK_KHR_external_semaphore_fd",
        );

        // 1. Vulkan 1.3 modern GPU with both platform extensions -> only platform extensions needed.
        let vk13 = negotiate_interop_extensions(
            vk::API_VERSION_1_3,
            false,
            vk::API_VERSION_1_3,
            |ext| ext == mem_plat || ext == sem_plat,
        );
        assert!(vk13.external_interop);
        assert_eq!(vk13.extensions, vec![mem_plat, sem_plat]);

        // 2. Vulkan 1.1 device -> adds `VK_KHR_timeline_semaphore` alongside platform extensions.
        let vk11 = negotiate_interop_extensions(
            vk::API_VERSION_1_1,
            false,
            vk::API_VERSION_1_1,
            |ext| ext == mem_plat || ext == sem_plat || ext == EXT_TIMELINE_SEMAPHORE,
        );
        assert!(vk11.external_interop);
        assert!(vk11.extensions.contains(&mem_plat));
        assert!(vk11.extensions.contains(&sem_plat));
        assert!(vk11.extensions.contains(&EXT_TIMELINE_SEMAPHORE));
        assert!(!vk11.extensions.contains(&EXT_SEMAPHORE_VK10_PREREQ));

        // 3. Vulkan 1.0 device + `VK_KHR_get_physical_device_properties2` on instance -> adds all
        //    `< 1.1` and `< 1.2` base extensions without duplicates.
        let vk10_full = negotiate_interop_extensions(
            vk::API_VERSION_1_0,
            true,
            vk::API_VERSION_1_0,
            |ext| {
                ext == mem_plat
                    || ext == sem_plat
                    || ext == EXT_TIMELINE_SEMAPHORE
                    || ext == EXT_SEMAPHORE_VK10_PREREQ
                    || EXT_MEMORY_VK10_PREREQS.contains(&ext)
            },
        );
        assert!(vk10_full.external_interop);
        for req in EXT_MEMORY_VK10_PREREQS {
            assert!(vk10_full.extensions.contains(&req));
        }
        assert!(vk10_full.extensions.contains(&EXT_SEMAPHORE_VK10_PREREQ));
        assert!(vk10_full.extensions.contains(&EXT_TIMELINE_SEMAPHORE));
        assert!(vk10_full.extensions.contains(&mem_plat));
        assert!(vk10_full.extensions.contains(&sem_plat));

        // 4. Vulkan 1.0 instance WITHOUT `VK_KHR_get_physical_device_properties2` cannot query UUID,
        //    so `external_interop` must be false even when device extensions exist.
        let vk10_no_uuid = negotiate_interop_extensions(
            vk::API_VERSION_1_0,
            false,
            vk::API_VERSION_1_0,
            |_| true,
        );
        assert!(!vk10_no_uuid.external_interop);

        // 5. Partial adapter (external semaphore only, no external memory) -> enables semaphore
        //    for GL interop without failing `vkCreateDevice`, while `external_interop` stays false.
        let sem_only = negotiate_interop_extensions(
            vk::API_VERSION_1_3,
            false,
            vk::API_VERSION_1_3,
            |ext| ext == sem_plat,
        );
        assert!(!sem_only.external_interop);
        assert_eq!(sem_only.extensions, vec![sem_plat]);

        // 6. Non-interop software adapter -> zero extensions requested, `external_interop == false`.
        let none = negotiate_interop_extensions(
            vk::API_VERSION_1_3,
            false,
            vk::API_VERSION_1_3,
            |_| false,
        );
        assert!(!none.external_interop);
        assert!(none.extensions.is_empty());
    }

    /// Can wgpu itself see a decoded picture's PLANES, or must somebody copy them out?
    ///
    /// wgpu 30 has `TextureFormat::NV12` and `P010`, and `wgpu-hal`'s Vulkan backend maps them to
    /// `G8_B8R8_2PLANE_420_UNORM` and `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` (`conv.rs:80`),
    /// which are exactly the two formats a Vulkan Video decoder produces here. With them, a
    /// decoded image can be wrapped as a wgpu texture and read through
    /// `TextureAspect::Plane0`/`Plane1` views (R8Unorm/Rg8Unorm, R16Unorm/Rg16Unorm), so the
    /// YUV-to-working-space shader a host already owns can read the picture WHERE IT WAS DECODED.
    ///
    /// This records what this adapter actually reports, because the whole no-copy path depends on
    /// it and "wgpu supports the format" is a claim about a build, not about a GPU.
    #[test]
    fn what_the_adapter_says_about_planar_video_formats() {
        let Some(shared) = crate::shared_device() else {
            eprintln!("no GPU adapter on this machine; nothing to report");
            return;
        };
        // The ADAPTER's answer is "what this GPU could do"; the DEVICE's is "what was actually
        // enabled". Only the second one decides whether a wrap succeeds at runtime, and they are
        // different questions - this crate asks for every stable feature the adapter reports, but
        // that intersection is a fact to read back, not to assume.
        let can = shared.adapter.features();
        let have = shared.device.features();
        let nv12 = (
            can.contains(wgpu::Features::TEXTURE_FORMAT_NV12),
            have.contains(wgpu::Features::TEXTURE_FORMAT_NV12),
        );
        let p010 = (
            can.contains(wgpu::Features::TEXTURE_FORMAT_P010),
            have.contains(wgpu::Features::TEXTURE_FORMAT_P010),
        );
        eprintln!(
            "[planar-formats] {}: NV12 adapter {} device {}, P010 adapter {} device {}",
            shared.adapter.get_info().name,
            nv12.0,
            nv12.1,
            p010.0,
            p010.1
        );
        // An adapter without these formats is a real configuration and costs a copy, not a
        // failure - so the assertion is the IMPLICATION, not the capability: whatever the adapter
        // can do, the shared device must have asked for, because this crate's whole contract is
        // that one device serves every consumer's fast path.
        assert_eq!(
            nv12.0, nv12.1,
            "the adapter supports NV12 but the shared device did not enable it: a consumer's \
             zero-copy video path would fail on a device that could have carried it"
        );
        assert_eq!(
            p010.0, p010.1,
            "the adapter supports P010 but the shared device did not enable it"
        );
    }

    /// Adopting a hand-built device must not break ordinary wgpu work on it.
    #[test]
    fn the_shared_device_still_computes() {
        let Some(shared) = crate::shared_device() else {
            eprintln!("no GPU adapter on this machine; nothing to compute");
            return;
        };
        let (device, queue) = (&shared.device, &shared.queue);
        let n = 256usize;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shared device smoke"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> v: array<f32>;\n\
                 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
                     v[id.x] = v[id.x] * 2.0;\n\
                 }"
                .into(),
            ),
        });
        let bytes = (n * std::mem::size_of::<f32>()) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&src));
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((n / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&buf, 0, &readback, 0, bytes);
        let _copy = crate::submit(queue, "gpu-info test", [enc.finish()]);
        crate::map_read(device, &readback.slice(..)).expect("map");
        let view = readback.slice(..).get_mapped_range().expect("map");
        let got: &[f32] = bytemuck::cast_slice(&view);
        assert!(
            got.iter().zip(&src).all(|(g, s)| *g == s * 2.0),
            "the adopted device computed the wrong values"
        );
    }
}
