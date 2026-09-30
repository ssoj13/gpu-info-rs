//! Buffers shared between wgpu devices through Vulkan external memory (`OPAQUE_WIN32`).
//!
//! **What:** one allocation that is a `wgpu::Buffer` on two devices of the same physical GPU - an
//! application's and a plug-in's, each with its own `VkInstance` and queue. The exporter makes it
//! ([`SharedBuffer::export`]) and hands out an [`Export`] (NT handle, allocation size, memory type);
//! the importer adopts it ([`SharedBuffer::import`]). Nothing is copied through host memory.
//!
//! **Why:** a plug-in that computes on its own device (the fractal: [`crate::compute_device`]) had to
//! read its Output back to host memory for the application to upload it again - 126 MB each way per
//! 2160x3840 RGBA32F frame, the largest transfer of Playa's playback (Nsight Systems, 2026-09-29).
//!
//! **Rules the Vulkan specification sets, kept here once:** an `OPAQUE_WIN32` import allocates the
//! exporter's size from the exporter's memory type ([`Export::memory_type`]; same physical device, so
//! the same memory-type list - [`device_uuid`] tells), dedicated as the export was; between the two
//! devices the buffer is released to and acquired from `VK_QUEUE_FAMILY_EXTERNAL`
//! ([`SharedBuffer::acquire`] / [`SharedBuffer::release`]). Synchronisation is the caller's
//! protocol (the OpenFX Vulkan site's: each side proves its own work complete before the other
//! touches the buffer).

use ash::vk;
use std::os::windows::io::{AsRawHandle, BorrowedHandle, FromRawHandle, OwnedHandle};
use std::sync::Arc;

/// The handle type of every shared allocation here.
const HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;

/// The Vulkan usage of a shared buffer, matching [`WGPU_USAGE`].
const VK_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// The wgpu usage of a shared buffer: written or read by shaders, copied either way.
pub const WGPU_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Why a buffer could not be shared.
#[derive(Debug, thiserror::Error)]
pub enum ExternalError {
    /// The device is not a Vulkan device.
    #[error("not a Vulkan device")]
    NotVulkan,
    /// The device was created without the named extension.
    #[error("the Vulkan device lacks {0}")]
    MissingExtension(&'static str),
    /// No device-local memory type accepts the buffer.
    #[error("no device-local memory type accepts a shared buffer (type bits {0:#x})")]
    NoMemoryType(u32),
    /// A Vulkan call failed.
    #[error("{call}: {result}")]
    Vulkan {
        /// The Vulkan entry point.
        call: &'static str,
        /// What it returned.
        result: String,
    },
    /// wgpu refused the raw encoder the ownership barrier is recorded in.
    #[error("the encoder is not an open Vulkan command buffer")]
    Encoder,
}

fn vk_error(call: &'static str) -> impl FnOnce(vk::Result) -> ExternalError {
    move |result| ExternalError::Vulkan {
        call,
        result: format!("{result:?}"),
    }
}

/// What the importer needs of an export: the NT handle and the allocation's size and memory type.
#[derive(Debug)]
pub struct Export {
    /// The NT handle of the allocation (closed when dropped; the importer does not take it).
    pub handle: OwnedHandle,
    /// The allocation's size in bytes (at least the buffer's).
    pub allocation: u64,
    /// The index of the exporter's memory type.
    pub memory_type: u32,
}

/// The Vulkan objects of one side, destroyed when wgpu drops its hal buffer.
struct Owned {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: both were created on `device` (still alive: wgpu drops its hal buffer before its device) and are
        // used by nothing once wgpu has dropped the hal buffer that referenced them.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// One side's view of a shared allocation: a `wgpu::Buffer` of its device.
pub struct SharedBuffer {
    buffer: wgpu::Buffer,
    raw: vk::Buffer,
    device: ash::Device,
    /// The queue family wgpu submits to on this device.
    family: u32,
}

impl std::fmt::Debug for SharedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedBuffer")
            .field("size", &self.buffer.size())
            .finish_non_exhaustive()
    }
}

/// The raw handles of a wgpu Vulkan device this module needs.
struct Raw {
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    family: u32,
}

fn raw(device: &wgpu::Device) -> Result<Raw, ExternalError> {
    // SAFETY: the hal device is only read: handles and function tables are cloned, nothing is destroyed; the objects
    // live as long as `device`, which every holder of the clones also keeps (the wgpu buffer).
    let hal =
        unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or(ExternalError::NotVulkan)?;
    if !hal
        .enabled_device_extensions()
        .contains(&ash::khr::external_memory_win32::NAME)
    {
        return Err(ExternalError::MissingExtension(
            "VK_KHR_external_memory_win32",
        ));
    }
    Ok(Raw {
        instance: hal.shared_instance().raw_instance().clone(),
        device: hal.raw_device().clone(),
        physical: hal.raw_physical_device(),
        family: hal.queue_family_index(),
    })
}

/// `VkPhysicalDeviceIDProperties::deviceUUID` of `device`'s GPU: two devices share memory only if
/// these match.
pub fn device_uuid(device: &wgpu::Device) -> Option<[u8; 16]> {
    // SAFETY: read-only access to the hal device (see `raw`).
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let instance = hal.shared_instance().raw_instance().clone();
    let physical = hal.raw_physical_device();
    drop(hal);
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    // SAFETY: a physical device of this instance; wgpu's Vulkan instances are 1.1+ (the 1.1 entry point is loaded).
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    Some(id.device_uuid)
}

impl SharedBuffer {
    /// A new `size`-byte shared buffer of `device` on a dedicated device-local allocation, and its [`Export`].
    pub fn export(device: &wgpu::Device, size: u64) -> Result<(Self, Export), ExternalError> {
        let raw = raw(device)?;
        let buffer = create_buffer(&raw.device, size)?;
        let (requirements, memory_type) = match get_requirements(buffer, &raw) {
            Ok(found) => found,
            Err(error) => {
                // SAFETY: created above, used by nothing yet.
                unsafe { raw.device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(HANDLE);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        let memory = allocate_and_bind(&raw.device, buffer, &info)?;
        let owned = Arc::new(Owned {
            device: raw.device.clone(),
            buffer,
            memory,
        });
        let win32 = ash::khr::external_memory_win32::Device::new(&raw.instance, &raw.device);
        let get = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(HANDLE);
        // SAFETY: the memory was allocated exportable as OPAQUE_WIN32 and the extension is enabled (checked in `raw`).
        let handle = unsafe { win32.get_memory_win32_handle(&get) }
            .map_err(vk_error("vkGetMemoryWin32HandleKHR"))?;
        if handle == 0 {
            return Err(ExternalError::Vulkan {
                call: "vkGetMemoryWin32HandleKHR",
                result: "a null handle".to_owned(),
            });
        }
        // SAFETY: an OPAQUE_WIN32 export is a new NT handle owned by the caller, closed once by OwnedHandle.
        let handle = unsafe {
            OwnedHandle::from_raw_handle(std::ptr::without_provenance_mut(handle.cast_unsigned()))
        };
        let shared = Self::wrap(device, owned, &raw, size);
        Ok((
            shared,
            Export {
                handle,
                allocation: requirements.size,
                memory_type,
            },
        ))
    }

    /// The `size`-byte buffer of an [`Export`] made by another device of the same physical GPU, as a buffer of
    /// `device`. The handle stays the caller's: borrowed, never closed (an OpenFX plug-in gets it as a bare value from
    /// `kOfxImagePropData`, and the host keeps it open).
    ///
    /// # Safety
    ///
    /// `handle`, `allocation` and `memory_type` are those of a live [`SharedBuffer::export`] (or an equivalent
    /// Vulkan `OPAQUE_WIN32` export of a dedicated buffer allocation) on a device whose [`device_uuid`] is
    /// `device`'s, and `size` is at most the exported buffer's size.
    pub unsafe fn import(
        device: &wgpu::Device,
        handle: BorrowedHandle<'_>,
        allocation: u64,
        memory_type: u32,
        size: u64,
    ) -> Result<Self, ExternalError> {
        let raw = raw(device)?;
        let buffer = create_buffer(&raw.device, size)?;
        let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
            .handle_type(HANDLE)
            .handle(handle.as_raw_handle() as isize);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation)
            .memory_type_index(memory_type)
            .push_next(&mut import)
            .push_next(&mut dedicated);
        let memory = allocate_and_bind(&raw.device, buffer, &info)?;
        let owned = Arc::new(Owned {
            device: raw.device.clone(),
            buffer,
            memory,
        });
        Ok(Self::wrap(device, owned, &raw, size))
    }

    /// `owned`'s buffer as a `wgpu::Buffer` of `device`, keeping `owned` until wgpu drops it.
    fn wrap(device: &wgpu::Device, owned: Arc<Owned>, raw: &Raw, size: u64) -> Self {
        let buffer = owned.buffer;
        // SAFETY: the VkBuffer outlives the hal buffer: the callback holds `owned` until wgpu destroys its hal
        // buffer; wgpu neither destroys an externally owned buffer nor maps it (no MAP usage).
        let hal = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                buffer,
                Box::new(move || drop(owned)),
            )
        };
        // SAFETY: a buffer of `device`'s VkDevice, bound to memory, `size` bytes (non-zero, the caller's), created
        // with the Vulkan usage of WGPU_USAGE.
        let wgpu_buffer = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                hal,
                &wgpu::BufferDescriptor {
                    label: Some("gpu-info shared buffer"),
                    size,
                    usage: WGPU_USAGE,
                    mapped_at_creation: false,
                },
            )
        };
        Self {
            buffer: wgpu_buffer,
            raw: buffer,
            device: raw.device.clone(),
            family: raw.family,
        }
    }

    /// The buffer on this side's device.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// A command buffer taking the buffer from `VK_QUEUE_FAMILY_EXTERNAL` to this device's queue: submitted before
    /// this side touches it, it makes the other side's completed writes visible here.
    pub fn acquire(&self, device: &wgpu::Device) -> Result<wgpu::CommandBuffer, ExternalError> {
        self.ownership(device, true)
    }

    /// A command buffer handing the buffer back to `VK_QUEUE_FAMILY_EXTERNAL`: submitted after this side's last
    /// access, it makes this side's writes visible to the other.
    pub fn release(&self, device: &wgpu::Device) -> Result<wgpu::CommandBuffer, ExternalError> {
        self.ownership(device, false)
    }

    fn ownership(
        &self,
        device: &wgpu::Device,
        acquire: bool,
    ) -> Result<wgpu::CommandBuffer, ExternalError> {
        let external = vk::QUEUE_FAMILY_EXTERNAL;
        let (from, to, src_stage, src_access, dst_stage, dst_access) = if acquire {
            (
                external,
                self.family,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            )
        } else {
            (
                self.family,
                external,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::empty(),
            )
        };
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(from)
            .dst_queue_family_index(to)
            .buffer(self.raw)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(if acquire {
                "gpu-info acquire from external"
            } else {
                "gpu-info release to external"
            }),
        });
        // SAFETY: the hal encoder is wgpu's open Vulkan command buffer of `device` (this buffer's VkDevice); only a
        // pipeline barrier over the live buffer is recorded, and nothing else goes into this encoder (wgpu 30 forbids
        // mixing raw and wgpu commands in one encoder).
        let recorded = unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
                hal.map(|hal| {
                    self.device.cmd_pipeline_barrier(
                        hal.raw_handle(),
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                })
            })
        };
        recorded.ok_or(ExternalError::Encoder)?;
        Ok(encoder.finish())
    }
}

/// A `size`-byte exportable/importable buffer (no memory yet).
fn create_buffer(device: &ash::Device, size: u64) -> Result<vk::Buffer, ExternalError> {
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE);
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(VK_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: a valid create info on a live device.
    unsafe { device.create_buffer(&info, None) }.map_err(vk_error("vkCreateBuffer"))
}

/// `buffer`'s requirements and the first device-local memory type that accepts it.
fn get_requirements(
    buffer: vk::Buffer,
    raw: &Raw,
) -> Result<(vk::MemoryRequirements, u32), ExternalError> {
    // SAFETY: `buffer` was created on this device.
    let requirements = unsafe { raw.device.get_buffer_memory_requirements(buffer) };
    // SAFETY: a physical device of this instance.
    let types = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical)
    };
    let found = types
        .memory_types
        .iter()
        .take(usize::try_from(types.memory_type_count).unwrap_or(0))
        .enumerate()
        .find(|(index, kind)| {
            requirements.memory_type_bits & (1 << index) != 0
                && kind
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .and_then(|(index, _)| u32::try_from(index).ok());
    found
        .map(|index| (requirements, index))
        .ok_or(ExternalError::NoMemoryType(requirements.memory_type_bits))
}

/// Allocate `info` and bind it to `buffer`, destroying the buffer (and the memory) on failure.
fn allocate_and_bind(
    device: &ash::Device,
    buffer: vk::Buffer,
    info: &vk::MemoryAllocateInfo<'_>,
) -> Result<vk::DeviceMemory, ExternalError> {
    // SAFETY: a valid allocate info on a live device.
    let memory = match unsafe { device.allocate_memory(info, None) } {
        Ok(memory) => memory,
        Err(result) => {
            // SAFETY: created by the caller, used by nothing yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(vk_error("vkAllocateMemory")(result));
        }
    };
    // SAFETY: buffer and memory of this device; the memory is dedicated to this buffer, offset 0.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: both unused.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(vk_error("vkBindBufferMemory")(result));
    }
    Ok(memory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::io::AsHandle;

    /// **Two devices of one GPU share a buffer.** Exported on the shared device (Playa's), imported on the
    /// compute device (a plug-in's: another `VkInstance`, another queue); the compute device writes a pattern
    /// between acquire and release, the shared device reads it back between its own acquire and release: the
    /// bytes are the pattern. A second pattern the other way proves the direction does not matter.
    #[test]
    #[ignore = "GPU: two devices of one adapter with VK_KHR_external_memory_win32"]
    fn two_devices_of_one_gpu_share_a_buffer() {
        let (Some(app), Some(plugin)) = (crate::shared_device(), crate::compute_device()) else {
            eprintln!("no GPU; skipped");
            return;
        };
        assert_eq!(
            device_uuid(&app.device),
            device_uuid(&plugin.device),
            "one physical GPU"
        );
        let size = 4096u64 * 16;
        let (exported, export) = SharedBuffer::export(&app.device, size).expect("export");
        // SAFETY: the export above, same GPU (asserted), the same size.
        let imported = unsafe {
            SharedBuffer::import(
                &plugin.device,
                export.handle.as_handle(),
                export.allocation,
                export.memory_type,
                size,
            )
        }
        .expect("import");
        let pattern = |seed: u32| -> Vec<u8> {
            (0..size as u32 / 4)
                .flat_map(|i| (i.wrapping_mul(2_654_435_761) ^ seed).to_ne_bytes())
                .collect()
        };
        // Write on `writer`'s side between acquire and release, then read on `reader`'s.
        let pass = |writer: (&wgpu::Device, &wgpu::Queue, &SharedBuffer),
                    reader: (&wgpu::Device, &wgpu::Queue, &SharedBuffer),
                    bytes: &[u8]| {
            let (device, queue, shared) = writer;
            queue.write_buffer(shared.buffer(), 0, bytes);
            let acquire = shared.acquire(device).expect("acquire");
            let release = shared.release(device).expect("release");
            crate::wait::wait(
                device,
                &crate::wait::submit(queue, "test write", [acquire, release]),
            )
            .expect("writer done");
            let (device, queue, shared) = reader;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut copy =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            copy.copy_buffer_to_buffer(shared.buffer(), 0, &staging, 0, size);
            let acquire = shared.acquire(device).expect("acquire");
            let release = shared.release(device).expect("release");
            crate::wait::wait(
                device,
                &crate::wait::submit(queue, "test read", [acquire, copy.finish(), release]),
            )
            .expect("reader done");
            crate::wait::map_read(device, &staging.slice(..)).expect("map");
            let got = staging
                .slice(..)
                .get_mapped_range()
                .expect("mapped")
                .to_vec();
            staging.unmap();
            got
        };
        let first = pattern(0x5a5a_1234);
        let got = pass(
            (&plugin.device, &plugin.queue, &imported),
            (&app.device, &app.queue, &exported),
            &first,
        );
        assert!(
            got == first,
            "the plug-in's writes are the application's bytes"
        );
        let second = pattern(0x0bad_f00d);
        let got = pass(
            (&app.device, &app.queue, &exported),
            (&plugin.device, &plugin.queue, &imported),
            &second,
        );
        assert!(got == second, "and the other way");
    }
}
