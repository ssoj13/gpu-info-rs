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

## Which module do I want?

| Need | Module | Cost per call | Notes |
| --- | --- | --- | --- |
| Adapter limits / features / formats | `gpu_info::query` | one-shot | needs wgpu |
| "How much VRAM does this box have" | `gpu_info::os` | ~1 s on macOS | shells out; probe once at start-up and cache |
| GPU graph in a UI, sampled at 1-10 Hz | `gpu_info::stats` | ~33 µs | syscall / sysfs only, never spawns |

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
