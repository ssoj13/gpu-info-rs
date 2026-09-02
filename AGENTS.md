# Integrating `gpu-info-rs` — guide for LLM coding agents

> **Pick the right module first.** This guide covers the **wgpu capability** half. Two other
> halves exist and need no wgpu at all:
>
> | Need | Module | Cost | Docs |
> | --- | --- | --- | --- |
> | adapter limits / features / formats | `gpu_info::query` | one-shot | this file |
> | "how much VRAM does this box have" | `gpu_info::os` | ~1 s on macOS (spawns) | `src/os.rs` |
> | live GPU graph, polled 1-10 Hz | `gpu_info::stats` | ~33 µs (no spawns) | `src/stats.rs` |
>
> For a monitor widget the answer is **always `stats`**, never `os`: `os` shells out to
> `system_profiler` / `nvidia-smi`, which hitches the caller's UI thread once per sample.
> `stats` is IOKit on macOS and DRM sysfs on Linux, and its module contract forbids process
> spawns — unavailable counters come back as `None` rather than as a spawn or a fake `0`.
> Take it with `default-features = false` and you pull no wgpu at all.

---


**Audience:** an AI coding assistant adding `wgpu-info-rs` to *another* Rust project
(e.g. `gitnexus-rs`, `vfx-rs`). Follow these steps literally. Do not improvise the API —
the exact signatures are listed at the bottom; use them verbatim.

The crate's purpose: query the GPU adapter's *real* capabilities so the host stops
hardcoding conservative limits (the classic `wgpu::Limits::default()` → "only 8 storage
buffers" trap).

---

## 0. Hard constraints — read first

1. **wgpu versions MUST match.** The public API takes and returns `wgpu` types
   (`wgpu::Adapter`, `wgpu::Limits`, `wgpu::Features`, `wgpu::Device`, `wgpu::Queue`).
   These types are **not** compatible across wgpu major versions, and Cargo will silently
   compile **two** copies of wgpu if versions differ, producing confusing
   "expected `wgpu::Adapter`, found `wgpu::Adapter`" errors.
   - Before integrating, find the host's wgpu version: search its `Cargo.toml`/lockfile for
     `wgpu = "…"`. This crate is built for **wgpu 29**. If the host is not on 29, STOP and
     tell the user — do not bump their wgpu without explicit approval.
2. **Spell wgpu types via the re-export** `wgpu_info::wgpu` at the integration site, so you
   are guaranteed to use the exact same `wgpu` the helper functions expect.
3. The crate is **not on crates.io** — depend on it by `path` or `git`.
4. The library is imported as `wgpu_info` (crate package name is `wgpu-info-rs`).
5. This is an additive change. Do **not** remove the host's existing wgpu dependency or
   refactor unrelated device setup. Touch only the limit/feature selection.

---

## 1. Decide what the host needs

Pick exactly one integration mode:

| Host goal | Use | Needs an existing `wgpu::Adapter`? |
| --- | --- | --- |
| "Use the GPU's real limits instead of `Limits::default()`" (most common) | `request_max_device` **or** `recommended_limits` | yes |
| "Print / log / export what this machine supports" | `query()` → `GpuReport` | no (creates its own instance) |
| "Detect capability regressions in CI" | `query()` + `GpuReport::diff` | no |

If unsure, the host almost always wants the **first** row (fix the limits).

---

## 2. Add the dependency

In the host crate's `Cargo.toml` (the crate that actually creates the `wgpu::Device`):

```toml
[dependencies]
# library only (no CLI binary / clap):
wgpu-info-rs = { path = "../wgpu-info-rs", default-features = false }
# git alternative:
# wgpu-info-rs = { git = "<repo-url>", default-features = false }
```

- In a Cargo **workspace**, prefer adding it to `[workspace.dependencies]` and referencing
  `wgpu-info-rs = { workspace = true }`, matching how the host manages `wgpu`.
- Keep `default-features = false` unless the host also wants the `wgpu-info` binary.

---

## 3A. Integration mode 1 — fix the limits (most common)

Find the device-creation site. Search the host for:
`request_device`, `Limits::default()`, `required_limits`, `DeviceDescriptor`.

### Option A — one-call helper (simplest)

Replace the whole `request_device` call:

```rust
use wgpu_info::wgpu; // same wgpu as wgpu-info-rs

// BEFORE:
let (device, queue) = adapter
    .request_device(&wgpu::DeviceDescriptor {
        label: Some("…"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(), // ← caps at 8 storage buffers
        ..Default::default()
    })
    .await?;

// AFTER:
let (device, queue) =
    wgpu_info::request_max_device(&adapter, wgpu::Features::empty()).await?;
```

- `extra_features` (2nd arg) is intersected with adapter support, so it can never make the
  request fail. Pass `wgpu::Features::empty()` for max limits only, or the specific features
  the host needs (e.g. `wgpu::Features::TEXTURE_BINDING_ARRAY`).
- Non-async caller? Use `wgpu_info::request_max_device_blocking(&adapter, features)?`.

### Option B — keep the host's `DeviceDescriptor`, swap only the limits

Use this when the host sets a custom `label`, `required_features`, memory hints, etc.:

```rust
required_limits: wgpu_info::recommended_limits(&adapter), // == adapter.limits()
```

That single line replaces `wgpu::Limits::default()` and nothing else.

### After the change

The resulting `device.limits()` now equals `adapter.limits()`. Any host code that was
hand-capped to the old defaults (e.g. bind groups packed to ≤ 8 storage buffers) can now use
the real limit — but **do not** rewrite that logic unless the user asks; just unblock it.

---

## 3B. Integration mode 2 — report / log capabilities

```rust
let report = wgpu_info::query(); // synchronous; enumerates all backends
// or restrict: wgpu_info::query_backends(wgpu_info::wgpu::Backends::VULKAN);

for a in &report.adapters {
    log::info!(
        "{} [{}] storage_buffers/stage={} max_buffer_size={}",
        a.name, a.backend,
        a.limits.max_storage_buffers_per_shader_stage,
        a.limits.max_buffer_size,
    );
}

// Human-readable dump:
println!("{}", report.to_pretty());
// JSON (GpuReport is serde Serialize/Deserialize):
let json = serde_json::to_string_pretty(&report)?;
```

`query()` builds its own throwaway `wgpu::Instance` — the host does not need to pass
anything. (It still links the same wgpu; the version rule in §0 applies.)

---

## 3C. Integration mode 3 — regression diff (CI)

```rust
let baseline: wgpu_info::GpuReport = serde_json::from_str(&saved_json)?;
let live = wgpu_info::query();
let diffs = baseline.diff(&live); // Vec<String>, empty == identical
if !diffs.is_empty() {
    for d in &diffs { eprintln!("cap change: {d}"); }
}
```

Or just run the bundled CLI: `wgpu-info --json > baseline.json`, then in CI
`wgpu-info --diff baseline.json` (exits non-zero on any difference). Note: `diff` compares
adapters positionally by index, so it assumes a stable enumeration order (same machine /
same requested backends).

---

## 4. Verify the integration

Run from the host project root (use the host's normal build flags/features):

```sh
cargo build
cargo test            # ensure nothing regressed
cargo clippy --all-targets -- -D warnings
```

Then confirm the limits actually changed at runtime (the whole point): log
`device.limits().max_storage_buffers_per_shader_stage` right after device creation and
verify it is far above 8 on a discrete GPU. Report the observed value to the user as
evidence — do not claim success without it.

---

## 5. Common errors & fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `expected struct wgpu::Adapter, found struct wgpu::Adapter` (two paths) | Host wgpu version ≠ this crate's | Align wgpu versions; use one `wgpu` in the workspace; spell types via `wgpu_info::wgpu`. |
| `cannot find function request_max_device` | Imported the package name | Import the **library** name `wgpu_info`, not `wgpu_info_rs` / `wgpu-info-rs`. |
| Pulls in `clap` you don't want | Default features on | `default-features = false`. |
| `request_device` fails on features | Requested unsupported features directly | Use `request_max_device` (it intersects with support) or `& adapter.features()`. |
| Async/`.await` in a sync context | `request_max_device` is async | Use `request_max_device_blocking`, or wrap in `pollster::block_on`. |

---

## 6. Full public API reference (wgpu 29)

```rust
// Re-export — always use this wgpu at the integration site.
pub use wgpu_info::wgpu;

// --- Querying (no adapter needed; creates its own instance, synchronous) ---
pub fn wgpu_info::query() -> GpuReport;
pub fn wgpu_info::query_backends(backends: wgpu::Backends) -> GpuReport;

// --- Device helpers (need the host's &wgpu::Adapter; versions must match) ---
pub fn wgpu_info::recommended_limits(adapter: &wgpu::Adapter) -> wgpu::Limits;

// Supported MSAA sample counts for a format (always includes 1), e.g. [1, 4]. Cross-platform
// (Metal/Vulkan/DX12/GL) via wgpu's format-feature flags. Intersect across every attachment a
// pass uses (color AND depth) before choosing a level.
pub fn wgpu_info::supported_sample_counts(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> Vec<u32>;

pub async fn wgpu_info::request_max_device(
    adapter: &wgpu::Adapter,
    extra_features: wgpu::Features,        // intersected with adapter support
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>;

pub fn wgpu_info::request_max_device_blocking(
    adapter: &wgpu::Adapter,
    extra_features: wgpu::Features,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>;

// --- Report types (all derive Serialize + Deserialize + Clone + PartialEq) ---
pub struct GpuReport {
    pub wgpu_info_version: String,
    pub wgpu_version: String,
    pub backends_requested: Vec<String>,
    pub adapters: Vec<AdapterReport>,
}
impl GpuReport {
    pub fn diff(&self, other: &GpuReport) -> Vec<String>; // "path: old -> new"
    pub fn to_pretty(&self) -> String;
}

pub struct AdapterReport {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor: u32,
    pub vendor_name: Option<String>,
    pub device: u32,
    pub pci_bus_id: String,
    pub driver: String,
    pub driver_info: String,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
    pub features: Vec<String>,        // wgpu::Features flag names
    pub limits: wgpu::Limits,         // embedded verbatim — full limit set
    pub downlevel: DownlevelReport,
    pub texture_formats: Vec<TextureFormatReport>,
}

pub struct DownlevelReport {
    pub is_webgpu_compliant: bool,
    pub shader_model: String,         // "Sm2" | "Sm4" | "Sm5"
    pub flags: Vec<String>,
}

pub struct TextureFormatReport {
    pub format: String,
    pub allowed_usages: Vec<String>,
    pub flags: Vec<String>,
    pub sample_counts: Vec<u32>,      // supported MSAA counts (incl. 1), e.g. [1, 4]
}
```

---

## 7. Worked example — gitnexus-rs

Host fact: `gitnexus-rs/crates/render-gpu/src/lib.rs` builds its device with
`required_limits: wgpu::Limits::default()`, and `crates/sim-scene` hand-packs bind groups to
stay within the 8-storage-buffer default.

Steps:
1. Confirm gitnexus-rs pins `wgpu = "29"` (it does, workspace-wide).
2. Add `wgpu-info-rs = { path = "../../wgpu-info-rs", default-features = false }` to
   `crates/render-gpu/Cargo.toml` (the device-owning crate).
3. In `render-gpu/src/lib.rs`, change the single line
   `required_limits: wgpu::Limits::default(),` →
   `required_limits: wgpu_info::recommended_limits(&adapter),`.
4. Build + run; log `device.limits().max_storage_buffers_per_shader_stage` and confirm it is
   the hardware value (hundreds of thousands on a discrete GPU), not 8.
5. Leave the `sim-scene` bind-group packing as-is unless the user asks to raise it; the
   limit is now unblocked and a separate change can take advantage of it.
