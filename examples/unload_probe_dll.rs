//! Fixture for the unload regression (`tests/unload_regression.rs`): a cdylib that negotiates
//! [`gpu_info::shared_device`] exactly as a plug-in does, so the probe (`unload_probe`) can load it,
//! call [`negotiate`] and unload it repeatedly. Not meant to be used on its own.
//!
//! Why it exists: the shared device is process-lifetime state (a never-dropped wgpu device and the
//! threads wgpu spawns for it). Created inside a module the host later unloads, that state outlived
//! its code: an OpenFX host crashed with an access violation in a wgpu thread running code of the
//! unloaded plug-in, and every reload leaked a whole device. `shared_device` now pins the module
//! that contains it; this fixture is how the regression is reproduced.

/// Negotiate the shared device; `1` when a device exists, `0` when none is available.
#[unsafe(no_mangle)]
pub extern "C" fn negotiate() -> i32 {
    i32::from(gpu_info::shared_device().is_some())
}
