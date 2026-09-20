//! The Vulkan half of the ONE shared device: a logical device that serves both wgpu compute and a
//! hardware video decoder, plus the raw `ash` handles a decoder needs.
//!
//! # Why this exists
//!
//! [`crate::shared_device`] used to be a plain `request_device`, so the process had one wgpu device
//! and a video decoder (ffmpeg-rs `av-hwaccel-vulkan`) had to open a SECOND `VkDevice` of its own.
//! Two devices cannot share an image: a decoded frame had to travel GPU -> RAM -> GPU to be
//! composited, on the playback path, every frame.
//!
//! wgpu never asks for the video extensions or for a decode queue, and it does not have to:
//! `wgpu_hal::vulkan::Adapter::open_with_callback` hands us the device-create info before
//! `vkCreateDevice`, so we add what the decoder needs to what wgpu requires, create the device
//! ONCE, and let wgpu adopt it (`Adapter::create_device_from_hal`). The decoder is then handed the
//! same `ash::Device` and its own queue family.
//!
//! Measured on this exact path before it was written (spike; RTX 3080 Ti, driver 616.64,
//! wgpu 30.0.1): an NV12 image created outside wgpu, filled through the DECODE queue, wrapped with
//! `texture_from_raw` + `create_texture_from_hal` and read by a compute pass - 0 wrong of 4096
//! pixels, with no host round trip.
//!
//! # What is NOT here
//!
//! Only Vulkan can do this today. On a Metal or DX12 adapter [`open_shared`] returns `None` and
//! the caller falls back to the ordinary `request_device` - loudly, because a silent fallback here
//! means every decoded frame is copied twice and nobody would ever see why.

use std::ffi::CStr;

use ash::vk;
use wgpu::hal::api::Vulkan;

/// Extensions the video decoder needs on top of whatever wgpu requires.
///
/// Both codec extensions are listed because the device is created once, at startup, before anyone
/// knows which clip will be opened. A codec whose extension the adapter lacks is simply dropped
/// from the request (see [`present_video_ext`]) rather than failing the whole negotiation.
const VIDEO_EXT: [&CStr; 4] = [
    c"VK_KHR_video_queue",
    c"VK_KHR_video_decode_queue",
    c"VK_KHR_video_decode_h265",
    c"VK_KHR_video_decode_h264",
];

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

/// The queue family wgpu renders on.
///
/// wgpu-hal 30 hard-codes it (`vulkan/adapter.rs`, `let family_index = 0; //TODO`) and takes
/// index 0 of it. A constant rather than a guess, and CHECKED after the device is open against
/// `Device::queue_family_index()`: if wgpu ever stops doing this, sharing must stop rather than
/// quietly hand a consumer the renderer's own queue.
const WGPU_FAMILY: u32 = 0;

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

/// Create the one logical device that serves wgpu AND a video decoder.
///
/// Returns `None` when this is not a Vulkan adapter, when it has no video-decode queue family, or
/// when the core video extensions are missing - the caller then negotiates an ordinary device and
/// says so. Returns an error only when the device could not be created at all, which is fatal
/// either way.
///
/// `features` and `limits` are what the caller would have requested anyway; this function adds
/// nothing to them. What it adds is the video extensions and one decode queue.
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

    let (video_ext, decode, compute, sync2_on, open) = unsafe {
        let hal = adapter.as_hal::<Vulkan>()?;
        let video_ext = present_video_ext(&hal);
        if !VIDEO_CORE.iter().all(|e| video_ext.contains(e)) {
            log::info!("gpu-info: adapter has no Vulkan video extensions; no shared decode device");
            return None;
        }
        let decode = decode_family(&hal)?;
        // wgpu's family and queue index, read from wgpu rather than inferred from the callback's
        // first create-info: `queue_family_index()`/`queue_index()` are public on wgpu-hal 30's
        // Vulkan device and are the authoritative pair. wgpu hard-codes family 0 today
        // (`adapter.rs`, `let family_index = 0; //TODO`), which is exactly the kind of fact that
        // must be read rather than assumed.
        let queue_families = families(&hal);
        // A decode queue that IS the renderer's queue is not a shared device, it is two
        // unsynchronised submitters wearing one handle. Refuse, loudly, rather than hand it over:
        // `vkQueueSubmit` is externally synchronised and nothing downstream could detect this.
        if decode == WGPU_FAMILY {
            log::warn!(
                "gpu-info: the only video-decode family ({decode}) is the one wgpu renders on; \
                 not sharing this device, because a decoder and the renderer would submit to one \
                 queue from two threads"
            );
            return None;
        }
        // A compute queue for whoever reads a decoded picture in place. A family of its own is
        // best; failing that, a SECOND queue in wgpu's family, which is ours because wgpu keeps
        // index 0; failing that, none - and a consumer is told so rather than handed the
        // renderer's queue.
        let compute = match compute_family(&queue_families, WGPU_FAMILY, Some(decode)) {
            Some(f) => Some((f, 0)),
            None => queue_families
                .get(WGPU_FAMILY as usize)
                .filter(|f| f.queue_count >= 2)
                .map(|_| (WGPU_FAMILY, 1)),
        };
        if compute.is_none() {
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
        if !sync2_on {
            log::warn!(
                "gpu-info: this adapter cannot enable synchronization2; a video decoder sharing \
                 this device would record barriers it does not support"
            );
        }
        let mut extra = video_ext.clone();
        if sync2_ext {
            extra.push(c"VK_KHR_synchronization2");
        }
        // Where a compute queue has to be a SECOND queue in wgpu's own family, the count is
        // raised by replacing that entry's priority slice - one entry per family is a hard rule
        // (VUID-VkDeviceCreateInfo-queueFamilyIndex-02802), so a second create-info for the same
        // family would be invalid usage rather than a second queue.
        let pair_in_main = compute == Some((WGPU_FAMILY, 1));
        let open = hal
            .open_with_callback(
                features,
                limits,
                &wgpu::MemoryHints::Performance,
                Some(Box::new(move |args| {
                    args.extensions.extend_from_slice(&extra);
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
                    want(decode);
                    if let Some((family, _)) = compute.filter(|(f, _)| *f != WGPU_FAMILY) {
                        want(family);
                    }
                    if pair_in_main {
                        // wgpu keeps index 0 of this entry (it passes its own pre-callback copy
                        // of the family index and the literal 0 to `device_from_raw`), so index 1
                        // is ours.
                        if let Some(first) = args.queue_create_infos.first_mut() {
                            *first = first.queue_priorities(&PRIORITY_PAIR);
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
        (video_ext, decode, compute, sync2_on, open)
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
            decode_queue: Some((decode, 0)),
            compute_queue: compute,
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
        }
    };

    // SAFETY: `open` was created from this adapter, and the descriptor asks for exactly the
    // features and limits it was opened with. Experimental features are enabled because CubeCL's
    // SPIR-V (and Metal MSL) passthrough needs that opt-in; no experimental FEATURE is requested.
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("gpu-info shared device (compute + video decode)"),
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

    log::info!(
        "gpu-info: shared Vulkan device: wgpu on {:?}, decode on {:?}, compute on {:?}, \
         video extensions {:?}",
        raw.main_queue,
        raw.decode_queue,
        raw.compute_queue,
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
        } else {
            assert!(
                shared.vulkan.is_none(),
                "this adapter cannot decode video, so no Vulkan half may be claimed"
            );
        }
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
        queue.submit([enc.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let view = readback.slice(..).get_mapped_range().expect("map");
        let got: &[f32] = bytemuck::cast_slice(&view);
        assert!(
            got.iter().zip(&src).all(|(g, s)| *g == s * 2.0),
            "the adopted device computed the wrong values"
        );
    }
}
