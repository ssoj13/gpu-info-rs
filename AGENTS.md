# Integrating `gpu-info-rs` — guide for LLM coding agents

Pool integration (2026-10-08): ResourcePool return/eviction drops retired
device resources outside its metadata mutex, with bounded retries and capacity
rechecks after reacquisition. The destructor-lock probe passed. Cargo.lock uses
latest compatible dependencies including wgpu 30.0.1. Final release library
tests: 49 passed, 15 separately ignored; the explicit compute-workspace GPU
reuse/dirty-clear/completion gate passed. Strict all-target/all-feature Clippy
passed. The OS RAM smoke checks its formula on one snapshot rather than
comparing independently changing live readings.

## VramWatch (2026-10-10)

`VramWatch::new(label, every)` (const) + `budget(GpuVramContext)` = the OS VRAM budget cached for `every`, failures
cached and logged once, retried after the interval. Feed it to `ResourcePool::set_budget` when a resource is
returned. One watch per device, keyed by `AdapterKey::of(adapter)` (backend, vendor, device, PCI bus, name; compute it once, not per call; it re-keys on another adapter, cached readings are never shared), recovery after a failure is logged once, the query must never call the watch. `budget_with(&AdapterKey, query)` is the test seam (`Duration::ZERO` / `MAX` for the clock). ofx-finish's `retained.rs` keeps an equivalent private copy
(`vram_quota`) that should move onto this.

`GpuLimits::tiling_reason_rows(w, h, row_bytes)`: the buffer-binding fit for padded rows (`tiling_reason` = whole-pixel rows).

## Run-time pool budgets (2026-10-09)

`ResourcePool::set_budget(bytes)` / `budget()`: the byte budget is an atomic read under the pool lock. Lowering
evicts the least recently returned idle resources at once (dropped outside the lock); 0 keeps nothing; raising
brings nothing back. A consumer whose idle quota follows a live figure (OS VRAM budget, user setting) calls
`set_budget` instead of keeping its own list. The const constructors are unchanged.

## Shared compute buffer workspace (2026-10-07)

`COMPUTE_BUFFERS` is exclusively for the exact canonical `compute_device()`:
512 MiB/256 idle entries, shared across effects. It is not an active-VRAM admission
budget and must never receive shared_device/application/external-device buffers.
`BufferWorkspace::new` checks device identity and falls back to ordinary allocation
on a foreign device. Create/use buffers inside device error scopes; after all submitted
uses complete, drop map views, unmap, close scopes successfully, then `recycle`.
No retained clone may be used afterward; no unsent encoder may reference it.
Dropping an incomplete/failed workspace never returns resources to the pool.

`zeroed_buffer` encodes its clear immediately in a same-device encoder, ordered before
first read/accumulation. `output_buffer` requires a full logical overwrite before read;
partial writers/padded FFT grids need explicit clear. `upload_slice` and
`overwrite_slice` use Pod bytes directly; uniform rewrites require earlier reads to
have completed. MAP_WRITE is refused. The pure accounting and ignored hardware tests
are added but unexecuted at this source checkpoint; no warm-allocation claim follows.

## Size RAM / VRAM budgets through `gpu_info::budget` (2026-10-01)

ONE home for the budget formulas (7 projects had 3 policies). RAM: `ram_budget_from(total, avail, policy, fraction,
reserve)` (pure), `ram_budget(policy, fraction, reserve)` (OS), `ram_budget_or(Some(bytes), ..)` (an explicit override
wins, no OS query). `Policy::Installed` = `min(f*total, total-reserve)` (scancache/playa: caches, property of the machine),
`Available` = `min(f*avail, avail-reserve)`, `Min` (default) = `min(f*total, total-reserve, avail-reserve)` (exv-tile
`usable_ram` = `Min, 0.5, 0`). OS query failed = `BudgetError::UnknownRam` / `UnknownVram`: never invent 2/8 GiB.
VRAM: `vram_share(unified)` 0.66 / 0.25, `vram_budget_from(headroom, unified)`, `vram_budget()` (os::query: free else
dedicated), `live_headroom(&VramQuerier)` (wgpu), `plan_vram(headroom, unified, fixed, one_tile) -> VramPlan{atlas, decode}`
(exv-tile semantics and numbers, pinned by tests). No floors in the core: a consumer applies its own (scancache `.max(1)`,
exv `MIN_BYTES`). Atomic live-resizable budget handles are NOT here (consumer / bufpool side).

## Reuse large GPU resources through `gpu_info::ResourcePool` (2026-09-29)

`ResourcePool<wgpu::Buffer>` / `ResourcePool<wgpu::Texture>` (`src/pool.rs`): one bounded pool per device and kind
(`const fn new(budget)`, so a `static` for a process-wide device); `take_buffer(device, BufferKey, label)` /
`take_texture(device, TextureKey, label)` take the most recently returned resource of the key or create one; `put`
returns one nobody else holds. Never create and drop a frame-sized buffer or texture per frame: on Windows the
`vkAllocateMemory` / `vkFreeMemory` behind it stall the process's GPU scheduling (Nsight Systems, Playa playback:
`vkFreeMemory` up to 668 ms, UI present waiting up to 379 ms). A pooled resource keeps old contents: clear what the
work reads before writing. Own types (a mapped staging slot) implement `Pooled`.

Bound the jobs that take from one pool with `gpu_info::Gate` (a counting semaphore: `enter` -> `Permit`,
`enter_owned(&Arc<Gate>)` -> `OwnedPermit` kept by a pending readback; cancellable workers use
`enter_until(|| !cancelled())` -> `Option<Permit>`, checking outside the mutex every 10 ms): at most as many jobs as the budget holds
resources, so an admitted job finds its resource pooled (unbounded, 18 render workers created and freed staging past
the budget). `GpuImage::from_pool` / `return_to` / `write_rgba_f32` keep `GpuImage`s in a `ResourcePool<Texture>`.

## Share a buffer between two wgpu devices with `gpu_info::external` (Windows, 2026-09-29)

`SharedBuffer::export(device, size)` -> (buffer, `Export { handle, allocation, memory_type }`); `SharedBuffer::import`
(unsafe) on another device of the same GPU (`device_uuid` equal) with the exporter's allocation and memory type -
the Vulkan rule for `OPAQUE_WIN32` - dedicated; `acquire` / `release` move it to and from `VK_QUEUE_FAMILY_EXTERNAL`
around each side's use; synchronisation is the caller's (each side proves its own work before the other touches it).
Both devices need `VK_KHR_external_memory_win32` (every stable wgpu feature asks for it). For the OpenFX Vulkan site
(ofx-rs plan0 3.A14 V): a plug-in on `compute_device` writing the application's frame in place.

## GPU work read back to the host runs on `compute_device()` (2026-09-28)

`gpu_info::compute_device()` is a second process-wide device (its own instance, adapter, max limits, stable features,
no video extensions). Effects that compute on the GPU and read the result back (ofx-rs `ofx::gpu_wgpu`,
`ofx-fractal`) use it, never `shared_device()`: one device has one FIFO queue, so their long dispatches held the UI's
frames (400 ms); across two devices the OS time-slices at workgroup boundaries (12 ms). Keep workgroups short on it
(16384 groups -> 12 ms, 1024 groups of the same total work -> 465 ms). Resources of one device are not usable on the
other.

## Submit and wait only through `gpu_info::{submit, wait, wait_idle, map_read}` (2026-09-28)

`let done = gpu_info::submit(&queue, "what it is", commands); gpu_info::wait(&device, &done)?` (the label feeds
`submit_stats()`) - never `queue.submit` +
`device.poll(Wait)`. A blocking `Device::poll` holds the device's snatch lock across the fence wait, so the UI's
`present` waited for the GPU; a timed one panics wgpu-core 30 (`device/resource.rs:948`) when another thread's
`submit`/`poll` retires the awaited submission between its fence read and its queue check. `submit` binds an
`on_submitted_work_done` callback to its submission (one process-wide lock); `wait` polls with `PollType::Poll` only
and sleeps on a condvar for `POLL_PERIOD` (1 ms; woken early by any thread's poll). `wait_idle(&device, &queue)`
waits for everything so far; `map_read(&device, &slice)` for a readback's map callback. `clippy.toml` here and in
ofx-rs / Playa disallows `wgpu::Device::poll` and `wgpu::Queue::submit`. Errors: one `WaitError`.

## The shared device enables `VK_KHR_external_semaphore_win32` where the adapter has it (2026-09-28)

`shared_vk::INTEROP_EXT` (Windows): added to the shared device like the video extensions (filtered by
`supports_extension`, recorded in `VulkanShared::device_extensions`). ofx-rs `ofx-host-wgpu` needs it to hand images
to OpenGL (the OpenFX OpenGL render site); no `wgpu::Features` bit asks for it. The hardware test asserts it is enabled
exactly when the adapter has it (mutant: not requested -> red).

## `shared_device()` pins its module (plug-ins, 2026-09-23)

The shared device is process-lifetime state: a `static` that is never dropped, plus the threads
wgpu runs for it. Before negotiating, `shared_device()` therefore pins the module that contains this
crate (`src/pin.rs`). For a plug-in DLL/.so this means: once it has used the shared device, the
host's `FreeLibrary`/`dlclose` no longer unmaps it. Every plug-in-level unload action still runs;
later loads reuse the same image and the same ONE device. Without the pin, an unloaded plug-in left
wgpu's threads running unmapped code (access violation) and leaked a whole device per reload. A
module that cannot be pinned gets `None`, with the reason logged as an error. Do not work around
this by creating your own device in a plug-in; that device would leak the same way. Regression:
`cargo test --test unload_regression -- --ignored` (Windows, GPU).

## Native Vulkan consumer boundary (2026-10-06)

Native Vulkan sharing and its `ash` dependency compile only for
`cfg(any(windows, target_os = "linux", target_os = "android", target_os = "freebsd"))`.
On other native targets, `shared_vk_unavailable.rs` keeps `VulkanShared` as an opaque API
placeholder and `SharedGpu::vulkan` is always `None`; macOS requests its ordinary native
Metal device. Do not add Vulkan fields to that placeholder or force MoltenVK for this path.

A consumer that accesses `decode_queue`, `entry`, other raw handles or wgpu's Vulkan HAL
must use this same compile-time predicate and keep a portable unavailable result elsewhere.
A runtime `Some` check alone cannot guard types/fields excluded from the target build.
FFmpeg's source fix `5f0e71c` follows GPU-info `35a81c9`: adoption returns `Ok(false)`,
the capability probe returns `false`, and Vulkan image wrapping/lending returns `ENOSYS`
on unsupported targets. Source review is not macOS compile/link/hardware acceptance.

## ONE device, and it can decode video (2026-09-19)

On a native Vulkan-capable target, `shared_device()` no longer just calls `request_device`.
On a Vulkan adapter it builds the logical
device through `wgpu_hal::vulkan::Adapter::open_with_callback`, ADDING the `VK_KHR_video_*`
extensions and a decode-queue family to what wgpu itself requires, and wgpu then adopts that device
(`create_device_from_hal`). `SharedGpu::vulkan` carries the raw `ash` handles (entry, instance,
physical device, device, queue families) for a consumer that speaks Vulkan — a hardware decoder
(ffmpeg-rs `av-hwaccel-vulkan`) runs on the SAME device, so a decoded image is used by a compute
pass with **no copy through host memory**. Proven before it was written: an NV12 image filled
through the decode queue, wrapped with `texture_from_raw`, read by a compute pass — 0 wrong of 4096
pixels (RTX 3080 Ti, driver 616.64, wgpu 30.0.1).

- `vulkan == None` means shared Vulkan video is unavailable: the target/backend is not Vulkan,
  or the adapter has no decode queue. This is normal for native Metal on macOS. Consumers
  must report that capability separately from ordinary wgpu rendering; no universal frame-cost
  claim follows from it.
- The device is created with `ExperimentalFeatures::enabled()` — required for passthrough shaders
  (CubeCL's SPIR-V / MSL). No experimental FEATURE is requested.
- Do NOT destroy anything in `SharedGpu::vulkan`: wgpu owns the device.
- `ash` is pinned to 0.38, the version wgpu-hal 30 uses. A consumer on another `ash` major would
  get a second, incompatible set of types.

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
     `wgpu = "…"`. This crate is built for **wgpu 30** (`Cargo.toml`). If the host is not on 30,
     STOP and tell the user — do not bump their wgpu without explicit approval.
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
