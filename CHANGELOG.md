# Changelog

All notable changes to `gpu-info-rs` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate is not
published to crates.io, so consumers pin it by git ref rather than by version.

## [Unreleased]

### Added

- **`readback::{map_read, block_on, ReadbackError}`: the one blocking buffer-readback wait.**
  `map_read(device, slice)` requests a read map, polls with `wait_indefinitely`, and waits for the
  callback on a mutex and condition variable (never a channel or `thread::park`, which reach
  `std::thread::current()` and on glibc pin a plug-in image to the host thread). A callback wgpu drops
  uncalled is `ReadbackError::Dropped`, not a hang. `GpuImage::read_rgba_f32` uses it (its error
  variants are unchanged); ofx-rs `ofx::gpu_wgpu` and `ofx-fractal` replace their own copies with it.
  Tests: `readback::tests::dropped_callback_reports_dropped`, and the ignored GPU
  `map_read_returns_written_bytes`.

### Changed

- **`shared_device()` honours wgpu's instance-level environment variables.** Its instance now comes
  from `shared_instance_descriptor()` = `InstanceDescriptor::new_without_display_handle_from_env()`,
  so `WGPU_BACKEND` (e.g. `dx12`, `vulkan`) selects the shared device's backend for the whole process,
  and the `InstanceFlags` and backend-option variables (`WGPU_VALIDATION`, `WGPU_DX12_COMPILER`, ...)
  apply. With none set the descriptor equals the previous one. The adapter request stays
  high-performance, non-fallback (`WGPU_POWER_PREF` / `WGPU_ADAPTER_NAME` are not read). Needed to
  run one consumer's GPU parity checks on DX12 as well as Vulkan. Regression:
  `tests/backend_env.rs` (ignored: needs a GPU) runs its own binary as children with
  `WGPU_BACKEND=dx12` and `=vulkan` and requires that backend (a backend without an adapter is
  reported SKIPPED); before the change the DX12 child got Vulkan. A unit test checks that without
  the variable the backend set is unchanged.

### Fixed

- **`shared_device()` pins the module that contains it before negotiating.** The shared device is
  process-lifetime state: a `static` that is never dropped, plus the threads wgpu runs for it. Linked
  into a dynamically loaded module (an OpenFX plug-in) that the host later unloaded, that state
  outlived its code. An OpenFX host crashed with `0xC0000005` at
  `<Unloaded_ofx_example_fractal.dll>+0xb9ef01` = `std::thread::Thread::park+0x71` on the thread
  "wgpu-hal WGL Instance Thread", which parks inside wgpu code for its instance's lifetime and
  returned into the unmapped image when it woke. Every reload also negotiated a new device in a new
  copy of the `static`: measured +230 MB private memory and +57 handles per load/unload cycle, and
  an OFX loaded-test process reached 20 GB and 4,409 handles. Now `GetModuleHandleExW(FROM_ADDRESS |
  PIN)` on Windows, and `dladdr` + `dlopen(RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE)` on Unix, keep the
  image mapped; later loads reuse it, so the negotiation stays ONE per process. A module that cannot
  be pinned gets no device (`None`, error logged with the reason). In an executable the pin is a
  no-op.
- Regression: `tests/unload_regression.rs` (ignored: needs a GPU, Windows) builds the
  `unload_probe_dll` cdylib fixture and the `unload_probe` example, which loads, negotiates and
  unloads the fixture 6 times. It requires the image to stay mapped, exactly one live thread starting
  inside it per cycle, and flat private memory and handles. Before the fix: the image unmapped every
  cycle, threads accumulated, and +1,190 MB / +386 handles over 6 cycles. After: 304 MB / 435 handles
  in every cycle.

### Known issues

See `TODO.md`: the GL backend's WGL thread (a separate decision), and two pre-existing gates
(macOS `cargo check` of `shared_vk`, strict rustdoc links), both present at `8035f5a` before this
change.

### Added

- **`VulkanShared` hands over QUEUES, not families.** `main_queue`, `decode_queue` and the new
  `compute_queue` are `(family, index)` pairs, because the index is what decides whose queue a
  handle is: `vkQueueSubmit` is externally synchronised, so a consumer that guessed index `0`
  could get the one wgpu is already submitting to from another thread. `compute_queue()` and
  `decode_queue()` fetch the `VkQueue` for their pair.
- **A compute queue is reserved for the decoder's in-place read.** The video-decode family cannot
  dispatch compute and the renderer's queue is not ours, so a consumer that wants to read a
  decoded picture where it lies needs a third queue. Preference: a compute family of its own,
  then a second queue in wgpu's family (wgpu keeps index 0, so index 1 is free, and the count is
  raised by replacing that entry's priority slice — one create-info per family is a hard rule,
  `VUID-VkDeviceCreateInfo-queueFamilyIndex-02802`), then none, said out loud. On an RTX 3080 Ti:
  wgpu `(0,0)`, decode `(3,0)`, compute `(2,0)`.
- **`device_api_version` beside `instance_api_version`, and `usable_api_version()` — the lower of
  the two.** That is the number a consumer must gate on.

### Fixed

- **The published API version was the loader's, not the device's.** wgpu fills
  `instance_api_version` from `vkEnumerateInstanceVersion` and then pins the application version
  to 1.3 whenever the loader is 1.1 or better. What decides whether a device entry point exists
  is the physical device's `apiVersion` — and ash fills an unloadable slot with a stub that
  PANICS at the first call, inside whichever library made it. A 1.3 loader in front of a 1.2
  driver would have passed every check and then panicked mid-decode.
- **`synchronization2` was reported as `true` unconditionally**, because this crate chains the
  feature struct on. A driver may ignore a chain entry it does not recognise and still create the
  device, so below 1.3 without `VK_KHR_synchronization2` the flag claimed a feature nothing had
  enabled — and every `2`-form barrier a decoder records against such a device is undefined
  behaviour. The struct is now chained only where the feature can exist, the extension is
  requested when it is the extension that provides it, and the flag is read back from the device.
- **wgpu's own queue family was inferred** from the first queue create-info through an
  `AtomicU32`. `Device::queue_family_index()` and `queue_index()` are public in wgpu-hal 30; they
  are read directly now, and the family wgpu hard-codes is verified rather than assumed — if it
  ever changes, sharing stops instead of quietly handing over the renderer's queue.
- **A video-decode family that IS wgpu's family was shared silently.** It is refused now: that is
  two unsynchronised submitters wearing one handle, and nothing downstream could detect it.
- **The `synchronization2` feature struct was `Box::leak`ed** once per call. wgpu-hal's callback
  signature (`CreateDeviceCallback<'this>`, `'this: 'pnext`) lets a local outlive
  `vkCreateDevice`, so it is a local now.
- `limits.rs` no longer converts `max_storage_buffer_binding_size` to `u64`; it is already `u64`
  in wgpu 30.

- **`gpu_info::stats` — live GPU counters cheap enough to poll from a UI frame loop.**
  `stats::query()` returns [`GpuStats`] with device utilisation, GPU-resident memory, total
  addressable memory and a `unified` flag.
  - **macOS**: IOKit `IOAccelerator` → `PerformanceStatistics` (`Device Utilization %`,
    `In use system memory`) plus `sysctlbyname("hw.memsize")`. Unprivileged — the same source
    Activity Monitor graphs. Measured **33 µs per call** on an Apple M4 Pro.
  - **Linux**: DRM sysfs (`gpu_busy_percent`, `mem_info_vram_used`, `mem_info_vram_total`)
    for `amdgpu` / `i915` / `xe`.
  - **Module contract: no process spawns, ever.** Anything that cannot be answered by a
    syscall or a sysfs read is reported as `None` instead of falling back to a spawn, so a
    caller can poll at 1-10 Hz without hitching its UI thread.
  - Requires no new dependencies and no wgpu: the IOKit and CoreFoundation entry points are
    declared locally, and all `unsafe` is confined to `stats/apple.rs` behind RAII wrappers.

### Changed

- `os.rs`: collapsed a nested `if` so `cargo clippy --all-targets -- -D warnings` passes
  again. No behaviour change.

### Notes for consumers

- Pick the right module: **`stats`** for monitors and per-frame polling, **`os`** for one-shot
  capability probes. `os::query()` shells out to `system_profiler` / `nvidia-smi` and costs
  roughly a second per call on macOS — correct for a start-up probe, wrong for a graph.
- NVIDIA utilisation is deliberately absent from `stats`: those counters live behind NVML,
  and `nvidia-smi` would violate the no-spawn contract. NVML can be added behind a feature
  without changing the public shape of [`GpuStats`].

[`GpuStats`]: https://docs.rs/gpu-info-rs/latest/gpu_info/stats/struct.GpuStats.html
