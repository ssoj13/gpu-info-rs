//! VK1 on real hardware (RTX 3080 Ti, NVIDIA driver >= 525): the compute device carries `VK_NV_optical_flow` and an
//! optical-flow queue apart from wgpu's family, and wgpu on it still computes. Ignored: needs an NVIDIA Turing+ GPU.
//!
//! Seeded mutant (the lead runs it once, then reverts): `compute_vk::open_compute`'s callback not pushing the queue
//! create info -> the device lacks the queue, `optical_flow` is `None` and the first `expect` fails.

#![cfg(all(feature = "wgpu", any(windows, target_os = "linux")))]

#[test]
#[ignore = "needs an NVIDIA Turing+ GPU with VK_NV_optical_flow"]
fn the_compute_device_has_the_optical_flow_queue_and_still_computes() {
    let gpu = gpu_info::compute_device().expect("a compute device");
    let of = gpu
        .optical_flow
        .expect("the RTX 3080 Ti reports VK_NV_optical_flow");
    let raw = gpu.raw.as_ref().expect("a Vulkan compute device");
    assert_ne!(of.family, raw.family, "a dedicated family (A5)");
    // SAFETY: the family and index were requested at vkCreateDevice (VK1); reading a queue handle is side-effect free.
    let queue = unsafe { raw.device.get_device_queue(of.family, of.index) };
    assert_ne!(queue, ash::vk::Queue::null());
    // Independent oracle: wgpu still runs on the device - 4 known words through a buffer round trip.
    let words = [1u32, 2, 0xdead_beef, 7];
    let (device, wq) = (&gpu.device, &gpu.queue);
    let bytes = std::mem::size_of_val(&words) as u64;
    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("optical flow test"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    wq.write_buffer(&src, 0, bytemuck::cast_slice(&words));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("optical flow test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&src, 0, &readback, 0, bytes);
    let _copy = gpu_info::submit(wq, "optical flow test", [encoder.finish()]);
    gpu_info::map_read(device, &readback.slice(..)).expect("map");
    let view = readback.slice(..).get_mapped_range().expect("mapped");
    let got: &[u32] = bytemuck::cast_slice(&view);
    assert_eq!(got, words);
}
