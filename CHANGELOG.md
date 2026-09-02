# Changelog

All notable changes to `gpu-info-rs` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate is not
published to crates.io, so consumers pin it by git ref rather than by version.

## [Unreleased]

### Added

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
