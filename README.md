# wgpu-info-rs

Portable GPU capability reporting via [`wgpu`] 29 — for Windows (DX12 / Vulkan),
Linux (Vulkan / GL) and macOS (Metal). Stop guessing conservative limits: ask the
adapter what it actually supports.

Ships both a **library** (`wgpu_info`) and a diagnostic **CLI** (`wgpu-info`).

> Integrating this crate into another project with an AI assistant?
> See [`LLM-INTEGRATION.md`](./LLM-INTEGRATION.md) for a precise, copy-paste recipe.

---

## Why this exists

A device built with `wgpu::Limits::default()` is capped at the conservative WebGPU
downlevel profile — e.g. `max_storage_buffers_per_shader_stage = 8`. Code then hand-packs
bind groups to stay under a limit that real hardware exceeds by orders of magnitude.

Measured on an RTX 3080 Ti with this tool:

| backend | `max_storage_buffers_per_shader_stage` | `Limits::default()` |
| ------- | -------------------------------------- | ------------------- |
| vulkan  | **524 288**                            | 8                   |
| dx12    | **262 144**                            | 8                   |
| gl      | 16                                     | 8                   |

`wgpu-info-rs` reports the real limits/features/formats and gives a one-call helper to
request a device that uses them.

---

## Install

Not published to crates.io. Depend on it by path or git, **pinned to the same wgpu major
as your project** (currently `29`).

```toml
# Path (sibling checkout)
wgpu-info-rs = { path = "../wgpu-info-rs", default-features = false }

# Git
wgpu-info-rs = { git = "https://…/wgpu-info-rs", default-features = false }
```

- `default-features = false` drops the `clap` dependency (the CLI binary). Keep defaults
  if you also want the `wgpu-info` binary.
- The crate name is `wgpu-info-rs`; the **library** is imported as `wgpu_info`.

> ⚠️ **Version coupling.** The public API exposes `wgpu` types (`wgpu::Limits`,
> `wgpu::Adapter`, `wgpu::Features`, `wgpu::Device`/`Queue`). Your project and this crate
> must resolve to the **same wgpu version**, or those types are incompatible. Pin wgpu
> identically, and spell the types via the re-export `wgpu_info::wgpu` to be safe.

---

## CLI

```sh
wgpu-info                  # human-readable report for every adapter / backend
wgpu-info --json           # full report as JSON
wgpu-info --backend vulkan # one backend: vulkan|dx12|metal|gl|primary|all
wgpu-info --adapter 0      # only the adapter at index 0
wgpu-info --json > caps.json
wgpu-info --diff caps.json # compare live machine vs saved report
```

`--diff` exits `0` when there are no differences and **non-zero** otherwise, so it can gate
CI / regression checks.

---

## Library

### Report the system

```rust
let report = wgpu_info::query(); // all backends; synchronous
for a in &report.adapters {
    println!(
        "{} ({}): {} storage buffers/stage, max_buffer_size = {}",
        a.name, a.backend,
        a.limits.max_storage_buffers_per_shader_stage,
        a.limits.max_buffer_size,
    );
}
// Restrict backends: wgpu_info::query_backends(wgpu_info::wgpu::Backends::VULKAN);
```

`GpuReport` / `AdapterReport` are `Serialize`/`Deserialize` and embed `wgpu::Limits`
verbatim, so every limit is covered with no hand-maintained mirror.

### Fix the "8 buffers" problem at the device-creation site

```rust
use wgpu_info::wgpu;

// Before:
//   required_limits: wgpu::Limits::default(),   // caps at 8 storage buffers
// After — request the adapter's real maximums in one call:
let (device, queue) =
    wgpu_info::request_max_device(&adapter, wgpu::Features::empty()).await?;

// Or keep your own DeviceDescriptor and only swap the limits:
//   required_limits: wgpu_info::recommended_limits(&adapter),
```

`request_max_device_blocking` is the non-async variant. `extra_features` is intersected
with what the adapter supports, so passing `wgpu::Features::all()` is safe (you get every
supported feature).

---

## Report schema

`GpuReport`:

| field                | type                  | meaning                                  |
| -------------------- | --------------------- | ---------------------------------------- |
| `wgpu_info_version`  | `String`              | this crate's version                     |
| `wgpu_version`       | `String`              | wgpu major it targets (`"29"`)           |
| `backends_requested` | `Vec<String>`         | backends asked for during enumeration    |
| `adapters`           | `Vec<AdapterReport>`  | one per enumerated adapter               |

`AdapterReport`:

| field                                | type                       | notes                                       |
| ------------------------------------ | -------------------------- | ------------------------------------------- |
| `name`                               | `String`                   | driver-reported adapter name                |
| `backend`                            | `String`                   | `vulkan` / `dx12` / `metal` / `gl` / …      |
| `device_type`                        | `String`                   | `DiscreteGpu` / `IntegratedGpu` / `Cpu` / … |
| `vendor`, `vendor_name`              | `u32`, `Option<String>`    | PCI vendor id + resolved name when known    |
| `device`                             | `u32`                      | backend device id                           |
| `pci_bus_id`                         | `String`                   | `bus:device.function` when available        |
| `driver`, `driver_info`              | `String`                   | driver name / version                       |
| `subgroup_min_size`, `_max_size`     | `u32`                      | wave / warp size range                      |
| `features`                           | `Vec<String>`              | supported `wgpu::Features` (flag names)     |
| `limits`                             | `wgpu::Limits`             | full limit set (embedded verbatim)          |
| `downlevel`                          | `DownlevelReport`          | sub-WebGPU capability flags + shader model  |
| `texture_formats`                    | `Vec<TextureFormatReport>` | per-format allowed usages / feature flags   |

`GpuReport::diff(&other) -> Vec<String>` lists field-level differences (`path: old -> new`).
`GpuReport::to_pretty() -> String` renders the human-readable report.

---

## Feature flags

| feature | default | effect                                       |
| ------- | ------- | -------------------------------------------- |
| `cli`   | yes     | builds the `wgpu-info` binary (pulls `clap`) |

Library-only consumers should use `default-features = false`.

---

## Platforms & backends

`query()` enumerates `wgpu::Backends::all()`. A single physical GPU may appear more than
once (e.g. under both Vulkan and DX12 on Windows). Edition 2024; pure Rust; no build
scripts. Tested against wgpu 29.0.3.

---

## License

MIT OR Apache-2.0.

[`wgpu`]: https://crates.io/crates/wgpu
