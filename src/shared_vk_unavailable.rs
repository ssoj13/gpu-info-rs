//! Portable API for native targets where wgpu has no Vulkan HAL backend.
//!
//! The real implementation is compiled on Windows, Linux, Android, and FreeBSD. Keeping this
//! module and [`VulkanShared`] available elsewhere makes [`crate::SharedGpu::vulkan`] one portable
//! API: it is always `None` when the target's native wgpu backend is not Vulkan-capable.

/// Raw handles for a Vulkan-backed shared device.
///
/// This type cannot be constructed on this target because wgpu has no Vulkan HAL backend here.
/// It exists only to keep [`crate::SharedGpu::vulkan`] source-compatible across native targets.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VulkanShared {
    _unavailable: (),
}

/// No Vulkan HAL exists on this target, so the caller must use its ordinary native backend.
pub(crate) fn open_shared(
    _adapter: &wgpu::Adapter,
    _features: wgpu::Features,
    _limits: &wgpu::Limits,
) -> Option<(wgpu::Device, wgpu::Queue, VulkanShared)> {
    None
}
