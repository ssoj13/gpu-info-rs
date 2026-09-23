# TODO / known issues

## GL backend in the shared instance (open decision)

`shared_device()` creates its `wgpu::Instance` with the default backends, which include GL. On
Windows the WGL instance starts a thread, "wgpu-hal WGL Instance Thread"
(`wgpu-hal 30.0.1 src/gles/wgl.rs:376-445`), that owns a hidden window and parks in a channel
`recv()` for as long as the instance lives. The shared device itself is Vulkan (or DX12/Metal), so
the thread and its window serve nothing. It was the thread that executed an unloaded OpenFX
plug-in's code (see CHANGELOG, Unreleased, Fixed). The module pin now prevents that, so this is no
longer a crash, only a thread, a hidden window and slower negotiation.

Option: build the shared instance with `Backends::PRIMARY` (Vulkan | DX12 | Metal). Open questions
before doing it: an adopter that needs GL through the shared instance (e.g. a surface on a GL-only
machine) would lose it, and the choice of adapter on a machine whose only usable adapter is GL. Not a
substitute for the pin: without the pin, an unloaded module would still orphan the device, its
memory and its handles on every reload.

## Pre-existing gate failures (present at `8035f5a`, before the module pin)

- `cargo check --lib --target aarch64-apple-darwin` fails in `src/shared_vk.rs`: 6 errors
  (`unresolved import wgpu::hal::api::Vulkan`, `cannot find vulkan in hal`). The module is compiled
  for every non-wasm target, but wgpu-hal has no Vulkan backend on macOS. Gate it on the targets
  where wgpu-hal builds Vulkan, or add a macOS stub.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` fails with 5 errors: public `os` docs link the
  private items `win_mem`, `crate::win_mem` and `vram`; `shared_device` links
  `OnceLock::get_or_init` unqualified; `shared_vk` links `open_shared` out of scope. With
  `--document-private-items` there is also `stats/windows.rs:43` `super::apple`, which is out of scope
  on Windows.
