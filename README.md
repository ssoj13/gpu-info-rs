# gpu-info-rs

One-stop GPU information crate. Two complementary APIs in one place:

- **wgpu capability report** — query a GPU's real limits, features, downlevel
  flags and texture formats through [`wgpu`], so apps stop guessing conservative
  defaults. Plus a DXGI adapter **VRAM budget** ([`VramQuerier`]) when you have a
  context.
- **`gpu_info::os`** — VRAM + system RAM **without a GPU context**: zero `unsafe`,
  no wgpu, queried via safe OS interfaces (`nvidia-smi` / `reg query` / sysfs /
  `system_profiler`). The lightweight "how much VRAM does this box have" path.

(Was two crates — `wgpu-info-rs` + `gpu-mem` — now merged into one.)

## Quick start

```rust
// wgpu capability report
let report = gpu_info::query();
for a in &report.adapters {
    println!("{}: max_storage_buffers = {}", a.name, a.limits.max_storage_buffers_per_shader_stage);
}

// OS-level VRAM/RAM, no GPU context needed
if let Some(m) = gpu_info::os::query() { println!("VRAM total: {}", m.total); }
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
