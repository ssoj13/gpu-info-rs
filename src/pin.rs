//! Keep the module that owns the shared device mapped for the life of the process.
//!
//! [`crate::shared_device`] creates process-lifetime state: a device held in a `static` that is
//! never dropped, and the threads wgpu starts for it (on Windows the GL backend's instance thread,
//! which parks inside wgpu code for as long as its instance lives). When this crate is linked into
//! a dynamically loaded module, typically an OpenFX plug-in, the host may unload that module at any
//! time, but the device, its threads and its callbacks do not go with it. After the unload a parked
//! thread that wakes returns into unmapped code, and the process dies with an access violation (an
//! OpenFX host crashed exactly so: `<Unloaded_plugin.dll>+0x..` in `std::thread::Thread::park`, on
//! the thread "wgpu-hal WGL Instance Thread"). Each reload also negotiates a fresh device in a
//! fresh copy of the `static`, leaking the previous one (measured at ~230 MB and 57 handles per
//! cycle).
//!
//! `pin_containing_module` therefore pins the module before the device is created: its code
//! stays mapped until the process exits, later loads of the same module reuse the image (so the
//! `static`, and with it the ONE device, persists), and the host's unload still runs every
//! plug-in-level unload action; only the image stays. In an executable this is a no-op.

/// Why the module could not be pinned; [`crate::shared_device`] then returns `None` rather than
/// create process-lifetime state inside a module that may be unmapped under it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinError(pub(crate) String);

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An address inside this crate's code: whatever module contains it is the module to pin.
fn anchor() -> *const () {
    pin_containing_module as *const ()
}

/// Pin the module containing this crate so it is never unmapped (Windows).
///
/// `GetModuleHandleExW(FROM_ADDRESS | PIN)` resolves the module that contains [`anchor`] and marks
/// it permanently loaded: every later `FreeLibrary` of it becomes a reference-count no-op.
#[cfg(windows)]
pub(crate) fn pin_containing_module() -> Result<(), PinError> {
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExW,
    };
    use windows::core::PCWSTR;

    let mut module = HMODULE::default();
    // SAFETY: with FROM_ADDRESS the "name" argument is read only as an address inside a loaded
    // module (never dereferenced as a string), and `anchor()` is code of this module; `module` is a
    // live out-parameter of the declared type. The pinned handle needs no release by definition.
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            PCWSTR(anchor().cast()),
            &mut module,
        )
    }
    .map_err(|error| {
        PinError(format!(
            "GetModuleHandleExW(FROM_ADDRESS | PIN) failed: {error}"
        ))
    })
}

/// Pin the module containing this crate so it is never unmapped (Linux, macOS, other Unix).
///
/// `dladdr` names the object that contains [`anchor`]; `dlopen(name, RTLD_NOW | RTLD_NOLOAD |
/// RTLD_NODELETE)` then adds `RTLD_NODELETE` to the already loaded object without loading anything
/// new, so no later `dlclose` unmaps it. The returned handle is deliberately never closed.
#[cfg(unix)]
pub(crate) fn pin_containing_module() -> Result<(), PinError> {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    // SAFETY: `dladdr` only reads the address and fills `info`, a live, correctly sized buffer.
    let found = unsafe { libc::dladdr(anchor().cast(), info.as_mut_ptr()) };
    if found == 0 {
        return Err(PinError(
            "dladdr found no object containing gpu-info-rs".into(),
        ));
    }
    // SAFETY: `dladdr` succeeded, so it initialised `info`.
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return Err(PinError(
            "dladdr returned no file name for gpu-info-rs".into(),
        ));
    }
    // SAFETY: `dli_fname` is the NUL-terminated path of an object that is loaded (it contains the
    // running code), and RTLD_NOLOAD guarantees nothing new is loaded or initialised.
    let handle = unsafe {
        libc::dlopen(
            info.dli_fname,
            libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE,
        )
    };
    if handle.is_null() {
        // SAFETY: `dlerror` returns NULL or a NUL-terminated message owned by the loader.
        let reason = unsafe {
            let message = libc::dlerror();
            if message.is_null() {
                "no dlerror message".to_owned()
            } else {
                std::ffi::CStr::from_ptr(message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        return Err(PinError(format!(
            "dlopen(RTLD_NOLOAD | RTLD_NODELETE) failed: {reason}"
        )));
    }
    Ok(())
}

/// No dynamic modules to unload (e.g. wasm32): nothing to pin.
#[cfg(not(any(windows, unix)))]
pub(crate) fn pin_containing_module() -> Result<(), PinError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinning the module that contains the test executable succeeds and is idempotent (the
    /// shared device may already have pinned it).
    #[test]
    fn pinning_the_containing_module_succeeds_repeatedly() {
        assert_eq!(pin_containing_module(), Ok(()));
        assert_eq!(pin_containing_module(), Ok(()));
    }
}
