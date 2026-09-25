//! The displays' ICC profiles, from the OS (OCIO `SystemMonitors`: `SystemMonitor_windows.cpp`,
//! `SystemMonitor_macos.cpp`).
//!
//! wgpu-free. The OS calls are FFI, isolated here as [`crate::win_mem`] isolates its syscall;
//! consumers (vfx-ocio's `SystemMonitors`) stay free of `unsafe`.

use std::path::PathBuf;

/// One display and its ICC profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayProfile {
    /// Windows: `"DISPLAY1, <monitor name>"`; macOS: `"Display <n>"`.
    pub name: String,
    /// The profile file.
    pub icc_path: PathBuf,
}

/// Every active display that has an ICC profile. Empty where the OS has no such API (Linux).
pub fn icc_profiles() -> Vec<DisplayProfile> {
    platform()
}

#[cfg(windows)]
fn platform() -> Vec<DisplayProfile> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use windows::Win32::Graphics::Gdi::{
        CreateDCW, DISPLAY_DEVICE_ACTIVE, DISPLAY_DEVICE_ATTACHED_TO_DESKTOP, DISPLAY_DEVICEW,
        DeleteDC, EnumDisplayDevicesW,
    };
    use windows::Win32::UI::ColorSystem::GetICMProfileW;
    use windows::core::{PCWSTR, PWSTR};

    fn wide(w: &[u16]) -> String {
        let len = w.iter().position(|&c| c == 0).unwrap_or(w.len());
        OsString::from_wide(&w[..len])
            .to_string_lossy()
            .into_owned()
    }
    let new_device = || DISPLAY_DEVICEW {
        cb: std::mem::size_of::<DISPLAY_DEVICEW>() as u32,
        ..Default::default()
    };

    let mut out = Vec::new();
    for n in 0.. {
        let mut adapter = new_device();
        // SAFETY: `adapter` is a valid, correctly sized DISPLAY_DEVICEW the call fills.
        if !unsafe { EnumDisplayDevicesW(PCWSTR::null(), n, &mut adapter, 0) }.as_bool() {
            break;
        }
        let flags = adapter.StateFlags;
        if flags & DISPLAY_DEVICE_ACTIVE != DISPLAY_DEVICE_ACTIVE
            || flags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP != DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
        {
            continue;
        }
        let device = PCWSTR(adapter.DeviceName.as_ptr());
        // SAFETY: `device` points at the NUL-terminated name inside `adapter`, alive for the call.
        let hdc = unsafe { CreateDCW(PCWSTR::null(), device, PCWSTR::null(), None) };
        if hdc.is_invalid() {
            continue;
        }
        let mut monitor = new_device();
        let mut path = [0u16; 261];
        let mut len = 260u32; // MAX_PATH
        // SAFETY: `monitor` and `path` are valid buffers of the sizes passed; `hdc` is live.
        let found = unsafe {
            let _ = EnumDisplayDevicesW(device, 0, &mut monitor, 0);
            GetICMProfileW(hdc, &mut len, Some(PWSTR(path.as_mut_ptr()))).as_bool()
        };
        // SAFETY: `hdc` came from CreateDCW and is released once.
        let _ = unsafe { DeleteDC(hdc) };
        if found {
            // "\\.\DISPLAY1" -> "DISPLAY1, <monitor>", as OCIO names it.
            let adapter_name = wide(&adapter.DeviceName);
            out.push(DisplayProfile {
                name: format!(
                    "{}, {}",
                    adapter_name.trim_start_matches("\\\\.\\"),
                    wide(&monitor.DeviceString)
                ),
                icc_path: PathBuf::from(wide(&path)),
            });
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn platform() -> Vec<DisplayProfile> {
    let mut out = Vec::new();
    use std::ffi::CStr;
    use std::os::raw::c_void;

    // Core Graphics types
    type CGDirectDisplayID = u32;
    type CFStringRef = *const c_void;
    type CFURLRef = *const c_void;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGGetActiveDisplayList(
            max_displays: u32,
            active_displays: *mut CGDirectDisplayID,
            display_count: *mut u32,
        ) -> i32;
    }

    #[link(name = "ColorSync", kind = "framework")]
    unsafe extern "C" {
        fn ColorSyncProfileCopyURLForDisplay(display: CGDirectDisplayID) -> CFURLRef;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFURLCopyFileSystemPath(url: CFURLRef, style: i32) -> CFStringRef;
        fn CFStringGetCString(
            string: CFStringRef,
            buffer: *mut i8,
            buffer_size: i64,
            encoding: u32,
        ) -> bool;
        fn CFRelease(cf: *const c_void);
    }

    // These mirror Apple's CFString/CFURL constant names verbatim.
    #[allow(non_upper_case_globals)]
    const kCFStringEncodingUTF8: u32 = 0x08000100;
    #[allow(non_upper_case_globals)]
    const kCFURLPOSIXPathStyle: i32 = 0;

    // Get active displays
    let mut display_ids = [0u32; 16];
    let mut display_count = 0u32;

    let result =
        unsafe { CGGetActiveDisplayList(16, display_ids.as_mut_ptr(), &mut display_count) };

    if result != 0 {
        return out;
    }

    for i in 0..display_count as usize {
        let display_id = display_ids[i];

        // Get ICC profile URL
        let url = unsafe { ColorSyncProfileCopyURLForDisplay(display_id) };
        if url.is_null() {
            continue;
        }

        // Convert URL to path
        let path_str = unsafe { CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle) };
        unsafe {
            CFRelease(url);
        }

        if path_str.is_null() {
            continue;
        }

        // Convert CFString to Rust string
        let mut buffer = [0i8; 1024];
        let success = unsafe {
            CFStringGetCString(path_str, buffer.as_mut_ptr(), 1024, kCFStringEncodingUTF8)
        };

        unsafe {
            CFRelease(path_str as *const c_void);
        }

        if success {
            let c_str = unsafe { CStr::from_ptr(buffer.as_ptr()) };
            if let Ok(path) = c_str.to_str() {
                let name = format!("Display {}", i + 1);
                out.push(DisplayProfile {
                    name,
                    icc_path: PathBuf::from(path),
                });
            }
        }
    }
    out
}

#[cfg(not(any(windows, target_os = "macos")))]
fn platform() -> Vec<DisplayProfile> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every listed display has a name and a profile path (none on a headless machine).
    #[test]
    fn listed_displays_have_names_and_profiles() {
        for d in icc_profiles() {
            assert!(
                !d.name.is_empty() && !d.icc_path.as_os_str().is_empty(),
                "{d:?}"
            );
        }
    }
}
