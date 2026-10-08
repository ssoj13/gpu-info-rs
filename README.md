# gpu-info-rs

One-stop GPU information crate. Two complementary APIs in one place:

- **wgpu capability report** — query a GPU's real limits, features, downlevel
  flags and texture formats through [`wgpu`], so apps stop guessing conservative
  defaults. Plus a DXGI adapter **VRAM budget** ([`VramQuerier`]) when you have a
  context.
- **`gpu_info::os`** — VRAM + system RAM **without a GPU context**: zero `unsafe`,
  no wgpu, queried via safe OS interfaces (`nvidia-smi` / `reg query` / sysfs /
  `system_profiler`). The lightweight "how much VRAM does this box have" path.
- **`gpu_info::stats`** — **live** counters (utilisation, memory in use) cheap enough to
  poll from a frame loop: IOKit on macOS, DRM sysfs on Linux. No wgpu, and **no process
  spawns** — 33 µs per call on an Apple M4 Pro.

(Was two crates — `wgpu-info-rs` + `gpu-mem` — now merged into one.)

## Shared compute buffers

Effects using the canonical `compute_device()` can share `COMPUTE_BUFFERS` instead
of multiplying independent idle budgets. Its limit is 512 MiB/256 idle buffers;
active work still needs admission. `BufferWorkspace` holds a job's exact-size buffers
and recycles only after explicit completion, successful device error scopes and
unmapped readbacks. Dropping a failed/incomplete workspace does not recycle it.
Use explicit encoder clears for accumulators, full-overwrite output buffers for
complete writers, and typed uploads to avoid temporary byte vectors. See the
[ownership contract](AGENTS.md#shared-compute-buffer-workspace-2026-10-07).
This source checkpoint has no new compile/hardware/performance receipt.

## Which module do I want?

| Need | Module | Cost per call | Notes |
| --- | --- | --- | --- |
| Adapter limits / features / formats | `gpu_info::query` | one-shot | needs wgpu |
| "How much VRAM does this box have" | `gpu_info::os` | ~1 s on macOS | shells out; probe once at start-up and cache |
| GPU graph in a UI, sampled at 1-10 Hz | `gpu_info::stats` | ~33 µs | syscall / sysfs only, never spawns |
| "How much RAM / VRAM may my cache use" | `gpu_info::budget` | one OS query | `Policy::{Installed, Available, Min}`; typed error, never a made-up default |

Using `os` where you meant `stats` is the classic mistake: a `system_profiler` spawn per
sample hitches the caller's UI thread. `stats` exists precisely to make that impossible.

### Live counters by platform

| Platform | Utilisation | Memory in use | Source |
| --- | --- | --- | --- |
| macOS (Apple GPU) | yes | yes | IOKit `IOAccelerator` → `PerformanceStatistics`, unprivileged |
| Linux (AMD / Intel) | yes | yes | DRM sysfs `gpu_busy_percent`, `mem_info_vram_*` |
| NVIDIA (any OS) | no | no | needs NVML; `nvidia-smi` would break the no-spawn contract |
| Windows (other) | no | no | PDH / DXGI budget not wired up yet |

Absent counters are reported as `None`, never as `0` — a UI should render them as `—`.

## Shared device by native backend

With the `wgpu` feature, `gpu_info::shared_device()` supplies the process-wide device.
Native Vulkan sharing is available only on Windows, Linux, Android and FreeBSD. On other
native targets, `SharedGpu::vulkan` remains `None` and `VulkanShared` is an opaque type;
macOS uses wgpu's ordinary Metal backend. No MoltenVK installation is required for this path.

Consumers must compile Vulkan handle/HAL access behind the same target predicate:
`cfg(any(windows, target_os = "linux", target_os = "android", target_os = "freebsd"))`.
Checking `vulkan.is_some()` at runtime cannot make unsupported fields or Vulkan HAL types
exist at compile time. See [the consumer rules](AGENTS.md#native-vulkan-consumer-boundary-2026-10-06).

The macOS source fix is `35a81c9` (2026-10-06). FFmpeg's downstream bridge fix is
`5f0e71c`; Playa `a03156b` pins both. This documentation refresh adds no build/test evidence;
current macOS compilation, linking and hardware acceptance remain unverified.

## Quick start

```rust
// wgpu capability report
let report = gpu_info::query();
for a in &report.adapters {
    println!("{}: max_storage_buffers = {}", a.name, a.limits.max_storage_buffers_per_shader_stage);
}

// OS-level VRAM/RAM, no GPU context needed
if let Some(m) = gpu_info::os::query() { println!("VRAM total: {}", m.total); }

// Live counters - safe to call every frame
if let Some(s) = gpu_info::stats::query() {
    println!("{:?} {:?}% {:?} bytes in use", s.name, s.util_pct, s.mem_used_bytes);
}
```

## Use

```toml
gpu-info = { git = "ssh://git@github.com/ssoj13/gpu-info-rs.git", branch = "main", package = "gpu-info-rs", default-features = false }
```

`default-features = false` drops the diagnostic CLI (`clap`). The `gpu-info` bin is
behind the `cli` feature.

## Build

```
python bootstrap.py b
python bootstrap.py t
python bootstrap.py c
```
