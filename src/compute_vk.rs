//! The compute device's Vulkan opening (ofx-rs motion spec C.17, VK1): [`crate::compute_device`] is opened through
//! `wgpu_hal::vulkan::Adapter::open_with_callback`, which adds `VK_NV_optical_flow`,
//! `VkPhysicalDeviceOpticalFlowFeaturesNV { opticalFlow }` and one queue of a dedicated optical-flow family BEFORE
//! `vkCreateDevice` when the adapter reports all of them (queried, never assumed). The hardware optical-flow
//! estimator of ofx-rs `ofx-motion` (`vk_nv`) then runs on the SAME `VkDevice` the plug-ins' wgpu compute runs on: no
//! second device, no external memory between them. A device without them opens as before (`optical_flow: None`).
//!
//! The raw handles are read once here ([`VulkanHandles`]) so no consumer calls `as_hal` for them (ofx-motion `vk_nv`,
//! the vfx-compute native Vulkan executor R2). The wgpu features requested are the caller's (every stable feature the
//! adapter has), so `VULKAN_EXTERNAL_MEMORY_WIN32` (the Vulkan-site import, `external::SharedBuffer::import`) is
//! enabled exactly as by the former `request_max_device`.
//!
//! **Replaces** the plain `request_max_device` of `compute_device_status`; the pattern is [`crate::shared_vk`]'s
//! (queues and a feature struct added by the callback, the enabled extensions recorded inside it).
//!
//! **Trap:** the optical-flow queue is externally synchronised; its one submitter is ofx-motion `vk_nv`, which owns
//! its submit lock. wgpu never submits to it.

/// The optical-flow queue the compute device was opened with: its family and its index within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpticalFlow {
    pub family: u32,
    pub index: u32,
}

#[cfg(any(windows, target_os = "linux", target_os = "android", target_os = "freebsd"))]
pub use vulkan::*;

#[cfg(any(windows, target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod vulkan {
    use super::OpticalFlow;
    use crate::shared_vk::WGPU_FAMILY;
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    /// The raw Vulkan handles of the compute device, read once at opening: function tables and handles (cheap to
    /// clone). wgpu owns the objects and keeps them alive while the [`crate::ComputeGpu`] lives (the process); never
    /// destroy them.
    #[derive(Clone)]
    pub struct VulkanHandles {
        pub instance: ash::Instance,
        pub device: ash::Device,
        pub physical: vk::PhysicalDevice,
        /// The family wgpu submits to (`wgpu_hal::vulkan::Device::queue_family_index`); never submit to its queue.
        pub family: u32,
        /// The physical device's `VkPhysicalDeviceProperties::apiVersion`.
        pub api: u32,
    }

    impl core::fmt::Debug for VulkanHandles {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("VulkanHandles")
                .field("physical", &self.physical)
                .field("family", &self.family)
                .field("api", &self.api)
                .finish_non_exhaustive()
        }
    }

    /// The first queue family carrying `VK_QUEUE_OPTICAL_FLOW_BIT_NV` with at least one queue that is not
    /// `wgpu_family` (OPERATOR-APPROVAL A5: a family shared with wgpu is not opened in v1), `None` otherwise.
    pub fn optical_flow_family(families: &[vk::QueueFamilyProperties], wgpu_family: u32) -> Option<u32> {
        families
            .iter()
            .zip(0u32..)
            .find(|(family, index)| {
                *index != wgpu_family
                    && family.queue_count > 0
                    && family.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV)
            })
            .map(|(_, index)| index)
    }

    /// The optical-flow family to open: the extension and its feature reported, a family found, and Vulkan 1.3 (where
    /// `VK_KHR_synchronization2` / `VK_KHR_format_feature_flags2`, which the extension needs, are core; C.17).
    pub fn wants_optical_flow(extension: bool, feature: bool, family: Option<u32>, api: u32) -> Option<u32> {
        (extension && feature && api >= vk::API_VERSION_1_3)
            .then_some(family)
            .flatten()
    }

    /// One priority, `'static` so the pointer in `VkDeviceQueueCreateInfo` is valid when `vkCreateDevice` reads it.
    static PRIORITY: [f32; 1] = [1.0];

    /// Whether the adapter reports `opticalFlow` in `VkPhysicalDeviceOpticalFlowFeaturesNV` (the query needs an
    /// instance of Vulkan 1.1 or later; below it the answer is `false`).
    fn optical_flow_feature(hal: &wgpu::hal::vulkan::Adapter) -> bool {
        let instance = hal.shared_instance();
        if instance.instance_api_version() < vk::API_VERSION_1_1 {
            return false;
        }
        let mut of = vk::PhysicalDeviceOpticalFlowFeaturesNV::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut of);
        // SAFETY: the physical device belongs to this instance (Vulkan 1.1 checked above); the chain is two
        // initialised structures that outlive the call.
        unsafe {
            instance
                .raw_instance()
                .get_physical_device_features2(hal.raw_physical_device(), &mut features);
        }
        of.optical_flow == vk::TRUE
    }

    /// Open the compute device on `adapter` with `features` and `limits`: through `open_with_callback` with the
    /// optical-flow extension, feature and queue when the adapter reports them (module docs), on a Vulkan adapter;
    /// `crate::request_max_device` exactly as before on any other (DX12, Metal: `(.., None, None)`).
    pub(crate) fn open_compute(
        adapter: &wgpu::Adapter,
        features: wgpu::Features,
        limits: &wgpu::Limits,
    ) -> Result<(wgpu::Device, wgpu::Queue, Option<OpticalFlow>, Option<VulkanHandles>), String> {
        let refused = |error: &dyn core::fmt::Display| {
            format!("cannot create compute device on {}: {error}", adapter.get_info().name)
        };
        let enabled = std::sync::Arc::new(std::sync::Mutex::new(Vec::<std::ffi::CString>::new()));
        let writer = std::sync::Arc::clone(&enabled);
        // Borrowed by the callback, not leaked: wgpu-hal's `CreateDeviceCallback<'this>` with `'this: 'pnext` lets a
        // local outlive `vkCreateDevice` (the `shared_vk` pattern; plan A5's `Box::leak` is not needed).
        let mut of_features = vk::PhysicalDeviceOpticalFlowFeaturesNV::default().optical_flow(true);
        let of_ref = &mut of_features;
        // SAFETY: `as_hal` yields the adapter wgpu is using; it is only read, and the guard is dropped before the device
        // is adopted. `open_with_callback`'s contract: the callback adds extensions, structures and queues the adapter
        // reported (each queried above it), never removes.
        let opened = unsafe {
            let Some(hal) = adapter.as_hal::<Vulkan>() else {
                let (device, queue) =
                    pollster::block_on(crate::request_max_device(adapter, features)).map_err(|e| refused(&e))?;
                return Ok((device, queue, None, None));
            };
            let caps = hal.physical_device_capabilities();
            let api = caps.properties().api_version;
            let extension = caps.supports_extension(ash::nv::optical_flow::NAME);
            let families = hal
                .shared_instance()
                .raw_instance()
                .get_physical_device_queue_family_properties(hal.raw_physical_device());
            let family = wants_optical_flow(
                extension,
                extension && optical_flow_feature(&hal),
                optical_flow_family(&families, WGPU_FAMILY),
                api,
            );
            if family.is_none() && extension {
                log::info!(
                    "gpu-info: the compute adapter lists VK_NV_optical_flow but not its feature, a dedicated family \
                     or Vulkan 1.3; the compute device opens without the optical-flow queue"
                );
            }
            let open = hal
                .open_with_callback(
                    features,
                    limits,
                    &wgpu::MemoryHints::Performance,
                    Some(Box::new(move |args| {
                        if let Some(family) = family {
                            args.extensions.push(ash::nv::optical_flow::NAME);
                            args.queue_create_infos.push(
                                vk::DeviceQueueCreateInfo::default()
                                    .queue_family_index(family)
                                    .queue_priorities(&PRIORITY),
                            );
                            let taken = core::mem::take(args.create_info);
                            *args.create_info = taken.push_next(of_ref);
                        }
                        if let Ok(mut all) = writer.lock() {
                            *all = args.extensions.iter().map(|e| std::ffi::CString::from(*e)).collect();
                        }
                    })),
                )
                .map_err(|e| refused(&e))?;
            let wgpu_family = open.device.queue_family_index();
            let handles = VulkanHandles {
                instance: hal.shared_instance().raw_instance().clone(),
                device: open.device.raw_device().clone(),
                physical: hal.raw_physical_device(),
                family: wgpu_family,
                api,
            };
            // The extension really enabled (read from the create info), on a family that is not wgpu's: never
            // assumed from the request.
            let on = enabled
                .lock()
                .map(|all| all.iter().any(|e| e.as_c_str() == ash::nv::optical_flow::NAME))
                .unwrap_or(false);
            let optical_flow = match family {
                Some(family) if on && family != wgpu_family => Some(OpticalFlow { family, index: 0 }),
                Some(family) => {
                    log::warn!(
                        "gpu-info: the optical-flow queue (family {family}) was requested but the device does not \
                         carry it apart from wgpu's family {wgpu_family}; no optical-flow queue"
                    );
                    None
                }
                None => None,
            };
            (open, optical_flow, handles)
        };
        let (open, optical_flow, handles) = opened;
        // SAFETY: `open` was created from this adapter, and the descriptor asks for exactly the features and limits it
        // was opened with.
        let (device, queue) = unsafe {
            adapter.create_device_from_hal(
                open,
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-info compute device"),
                    required_features: features,
                    required_limits: limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                },
            )
        }
        .map_err(|e| refused(&e))?;
        log::info!(
            "gpu-info: compute device on {}: optical flow {:?}",
            adapter.get_info().name,
            optical_flow
        );
        Ok((device, queue, optical_flow, Some(handles)))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
            vk::QueueFamilyProperties {
                queue_flags: flags,
                queue_count: count,
                ..Default::default()
            }
        }

        /// The RTX 30 layout: graphics+compute (0), transfer (1), compute (2), video decode (3), optical flow (4).
        /// Independent oracle: the index of the only family carrying the bit, written here.
        #[test]
        fn the_dedicated_optical_flow_family_is_found() {
            let families = [
                family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 16),
                family(vk::QueueFlags::TRANSFER, 2),
                family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 8),
                family(vk::QueueFlags::VIDEO_DECODE_KHR, 1),
                family(vk::QueueFlags::OPTICAL_FLOW_NV, 1),
            ];
            assert_eq!(optical_flow_family(&families, 0), Some(4));
            assert_eq!(optical_flow_family(&families[..4], 0), None, "no family carries the bit");
            assert_eq!(
                optical_flow_family(&[family(vk::QueueFlags::OPTICAL_FLOW_NV, 0)], 1),
                None,
                "a family without queues"
            );
        }

        /// A5: the bit on wgpu's own family is not used (wgpu creates one queue there).
        #[test]
        fn the_wgpu_family_is_never_the_optical_flow_family() {
            let families = [family(vk::QueueFlags::COMPUTE | vk::QueueFlags::OPTICAL_FLOW_NV, 2)];
            assert_eq!(optical_flow_family(&families, 0), None);
        }

        /// Every condition is required; nothing is assumed (C.17 "queried, never assumed"). Seeded mutant (the lead
        /// runs it once, then reverts): `wants_optical_flow` ignoring `feature` -> FAIL on "feature missing".
        #[test]
        fn the_queue_is_wanted_only_when_everything_is_reported() {
            let v13 = vk::API_VERSION_1_3;
            assert_eq!(wants_optical_flow(true, true, Some(4), v13), Some(4));
            assert_eq!(wants_optical_flow(false, true, Some(4), v13), None, "extension missing");
            assert_eq!(wants_optical_flow(true, false, Some(4), v13), None, "feature missing");
            assert_eq!(wants_optical_flow(true, true, None, v13), None, "no family");
            assert_eq!(wants_optical_flow(true, true, Some(4), vk::API_VERSION_1_2), None, "Vulkan 1.2");
        }
    }
}
