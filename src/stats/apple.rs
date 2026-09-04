//! Apple GPU counters straight from the IORegistry — unprivileged, and spawn-free.
//!
//! # What this reads and why here
//!
//! The AGX driver publishes a `PerformanceStatistics` dictionary on its `IOAccelerator`
//! node. Those are the numbers Activity Monitor's GPU history graph is drawn from, they need
//! no entitlement, no root and no private framework, and reading them is a property lookup
//! rather than a process spawn. `powermetrics` needs root because it subscribes to channels
//! this module never touches — that is not a reason to avoid IOKit.
//!
//! Verified keys (Apple M4 Pro, macOS 15): `Device Utilization %`, `In use system memory`.
//! Cross-check with `ioreg -rc IOAccelerator`.
//!
//! # On the FFI
//!
//! Kernel counters have no pure-Rust route, so the C ABI is the floor here, exactly as it is
//! for the crate's Windows RAM syscall. The bindings are declared locally instead of pulled
//! from a generated-bindings crate: four IOKit entry points and a handful of CoreFoundation
//! accessors are a stable ABI, whereas a bindings crate adds a dependency whose API churns
//! faster than the framework it wraps. All `unsafe` is confined to this file behind the safe
//! RAII wrappers below, so nothing above [`query`] can leak or double-free a CF object.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

use super::GpuStats;

// ── C types ────────────────────────────────────────────────────────────────────────────

type CFTypeRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFTypeID = usize;
type CFIndex = isize;
/// `mach_port_t` / `io_object_t` — both are `u32` on every Apple platform.
type IoObject = u32;

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const K_CF_NUMBER_SINT64_TYPE: CFIndex = 4;
const K_CF_NUMBER_FLOAT64_TYPE: CFIndex = 6;
/// `kIOMainPortDefault` is documented as 0, which keeps `IOMainPort` out of the picture.
const K_IO_MAIN_PORT_DEFAULT: IoObject = 0;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
    /// Consumes one reference to `matching`, so the caller must not release it.
    fn IOServiceGetMatchingService(
        main_port: IoObject,
        matching: CFMutableDictionaryRef,
    ) -> IoObject;
    fn IORegistryEntryCreateCFProperty(
        entry: IoObject,
        key: CFStringRef,
        allocator: CFTypeRef,
        options: u32,
    ) -> CFTypeRef;
    fn IOObjectRelease(object: IoObject) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        alloc: CFTypeRef,
        c_str: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFStringGetCString(
        the_string: CFStringRef,
        buffer: *mut c_char,
        buffer_size: CFIndex,
        encoding: u32,
    ) -> bool;
    fn CFDictionaryGetValue(dict: CFTypeRef, key: *const c_void) -> CFTypeRef;
    fn CFNumberGetValue(number: CFTypeRef, the_type: CFIndex, value_ptr: *mut c_void) -> bool;
    fn CFDataGetLength(data: CFTypeRef) -> CFIndex;
    fn CFDataGetBytePtr(data: CFTypeRef) -> *const u8;
    fn CFGetTypeID(cf: CFTypeRef) -> CFTypeID;
    fn CFNumberGetTypeID() -> CFTypeID;
    fn CFStringGetTypeID() -> CFTypeID;
    fn CFDataGetTypeID() -> CFTypeID;
    fn CFDictionaryGetTypeID() -> CFTypeID;
    fn CFRelease(cf: CFTypeRef);
}

// libSystem is always linked; `hw.memsize` is the unified-memory total.
unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *const c_void,
        newlen: usize,
    ) -> i32;
}

// ── RAII wrappers ──────────────────────────────────────────────────────────────────────

/// A CF object obtained from a `Create`/`Copy` function, i.e. one we own and must release.
struct CfOwned(CFTypeRef);

impl CfOwned {
    /// # Safety
    /// `raw` must come from a CF `Create`/`Copy` call (create rule) or be null.
    unsafe fn from_create(raw: CFTypeRef) -> Option<Self> {
        (!raw.is_null()).then_some(Self(raw))
    }

    fn as_raw(&self) -> CFTypeRef {
        self.0
    }

    fn type_id(&self) -> CFTypeID {
        // SAFETY: `self.0` is a live CF object for as long as `self` exists.
        unsafe { CFGetTypeID(self.0) }
    }
}

impl Drop for CfOwned {
    fn drop(&mut self) {
        // SAFETY: created under the create rule and released exactly once.
        unsafe { CFRelease(self.0) };
    }
}

/// An owned `CFString`, used for property keys.
struct CfKey(CFStringRef);

impl CfKey {
    fn new(text: &str) -> Option<Self> {
        let c = CString::new(text).ok()?;
        // SAFETY: `c` outlives the call; CFString copies the bytes.
        let raw = unsafe {
            CFStringCreateWithCString(ptr::null(), c.as_ptr(), K_CF_STRING_ENCODING_UTF8)
        };
        (!raw.is_null()).then_some(Self(raw))
    }

    fn as_ptr(&self) -> CFStringRef {
        self.0
    }
}

impl Drop for CfKey {
    fn drop(&mut self) {
        // SAFETY: from `CFStringCreateWithCString` (create rule), released once.
        unsafe { CFRelease(self.0) };
    }
}

/// A matched IOService handle.
struct IoService(IoObject);

impl IoService {
    /// Matches the first service of `class`, e.g. `IOAccelerator`.
    fn matching(class: &str) -> Option<Self> {
        let name = CString::new(class).ok()?;
        // SAFETY: `name` outlives the call. `IOServiceGetMatchingService` consumes the
        // matching dictionary's reference, so it must not be released here.
        let service = unsafe {
            let matching = IOServiceMatching(name.as_ptr());
            if matching.is_null() {
                return None;
            }
            IOServiceGetMatchingService(K_IO_MAIN_PORT_DEFAULT, matching)
        };
        (service != 0).then_some(Self(service))
    }

    /// One property of this registry entry, or `None` when absent.
    fn property(&self, key: &str) -> Option<CfOwned> {
        let key = CfKey::new(key)?;
        // SAFETY: live service and key; the result follows the create rule.
        unsafe {
            let raw = IORegistryEntryCreateCFProperty(self.0, key.as_ptr(), ptr::null(), 0);
            CfOwned::from_create(raw)
        }
    }
}

impl Drop for IoService {
    fn drop(&mut self) {
        // SAFETY: `IOServiceGetMatchingService` hands over a reference we release once.
        unsafe { IOObjectRelease(self.0) };
    }
}

// ── CF decoding ────────────────────────────────────────────────────────────────────

/// Reads a numeric dictionary entry as `f64`.
///
/// The driver mixes integer and floating `OSNumber`s, so both widths are tried; a key that is
/// present but not a number yields `None` rather than a bogus zero.
fn dict_number(dict: &CfOwned, key: &str) -> Option<f64> {
    let key = CfKey::new(key)?;
    // SAFETY: `dict` is a live CFDictionary (the caller checks its type id), and
    // `CFDictionaryGetValue` follows the get rule, so the value is borrowed, not owned.
    unsafe {
        let value = CFDictionaryGetValue(dict.as_raw(), key.as_ptr().cast());
        if value.is_null() || CFGetTypeID(value) != CFNumberGetTypeID() {
            return None;
        }
        let mut as_i64: i64 = 0;
        if CFNumberGetValue(
            value,
            K_CF_NUMBER_SINT64_TYPE,
            (&raw mut as_i64).cast::<c_void>(),
        ) {
            return Some(as_i64 as f64);
        }
        let mut as_f64: f64 = 0.0;
        if CFNumberGetValue(
            value,
            K_CF_NUMBER_FLOAT64_TYPE,
            (&raw mut as_f64).cast::<c_void>(),
        ) {
            return Some(as_f64);
        }
        None
    }
}

/// Decodes a `CFString` or a `CFData`-wrapped C string — `IOAccelerator` uses both for names.
fn decode_text(value: &CfOwned) -> Option<String> {
    // SAFETY: `value` is a live CF object owned by `CfOwned`.
    unsafe {
        let id = value.type_id();
        if id == CFStringGetTypeID() {
            // Names here are SoC model strings; 256 bytes is far more than any of them need.
            let mut buf: [c_char; 256] = [0; 256];
            let ok = CFStringGetCString(
                value.as_raw(),
                buf.as_mut_ptr(),
                buf.len() as CFIndex,
                K_CF_STRING_ENCODING_UTF8,
            );
            if !ok {
                return None;
            }
            return CStr::from_ptr(buf.as_ptr())
                .to_str()
                .ok()
                .map(str::to_owned);
        }
        if id == CFDataGetTypeID() {
            let len = CFDataGetLength(value.as_raw());
            let ptr = CFDataGetBytePtr(value.as_raw());
            if len <= 0 || ptr.is_null() {
                return None;
            }
            let bytes = std::slice::from_raw_parts(ptr, len as usize);
            // The driver stores names as NUL-terminated C strings inside CFData.
            let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
            return std::str::from_utf8(&bytes[..end]).ok().map(str::to_owned);
        }
        None
    }
}

/// Total physical RAM via `sysctlbyname` — a syscall, not a `sysctl` process spawn.
fn sys_memsize() -> Option<u64> {
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `value` and `len` are valid for the requested size; no new value is written.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &raw mut len,
            ptr::null(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value)
}

// ── Query ──────────────────────────────────────────────────────────────────────────

/// Both spellings exist in the wild: Apple Silicon answers `IOAccelerator`, and some driver
/// generations only answer the concrete `AGXAccelerator` class.
const ACCELERATOR_CLASSES: [&str; 2] = ["IOAccelerator", "AGXAccelerator"];

pub(super) fn query() -> Option<GpuStats> {
    for class in ACCELERATOR_CLASSES {
        let Some(service) = IoService::matching(class) else {
            continue;
        };
        let Some(stats) = service.property("PerformanceStatistics") else {
            continue;
        };
        // A non-dictionary here means the driver shape changed; reading it as one anyway
        // would be undefined behaviour.
        // SAFETY: `stats` is a live CF object.
        if stats.type_id() != unsafe { CFDictionaryGetTypeID() } {
            continue;
        }

        let util =
            dict_number(&stats, "Device Utilization %").map(|v| (v as f32).clamp(0.0, 100.0));
        let used = dict_number(&stats, "In use system memory")
            // Negative or NaN would be a decode error, not a reading.
            .and_then(|v| (v.is_finite() && v >= 0.0).then_some(v as u64));
        if util.is_none() && used.is_none() {
            continue;
        }

        let name = service.property("model").as_ref().and_then(decode_text);
        let driver_class = service
            .property("IOClass")
            .as_ref()
            .and_then(decode_text)
            .unwrap_or_default();
        // Apple's own driver family (`AGXAccelerator…`) is always unified memory. A discrete
        // AMD GPU in an Intel Mac answers the same match with a different dictionary, and
        // calling that "unified" would misreport its VRAM as system RAM.
        let unified = driver_class.starts_with("AGX") || cfg!(target_arch = "aarch64");

        return Some(GpuStats {
            name,
            util_pct: util,
            mem_used_bytes: used,
            mem_total_bytes: if unified { sys_memsize() } else { None },
            unified,
            // IOKit PerformanceStatistics has no thermal, clock, power or fan counters.
            ..Default::default()
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memsize_is_sane() {
        let total = sys_memsize().expect("hw.memsize must answer on macOS");
        assert!(total >= 1 << 30, "suspiciously small RAM total: {total}");
    }

    #[test]
    fn accelerator_answers_on_a_mac() {
        let stats = query().expect("every Mac has an IOAccelerator node");
        eprintln!("apple gpu stats = {stats:?}");
        assert!(
            !stats.is_empty(),
            "matched the node but decoded no counter - key names may have changed"
        );
    }

    #[test]
    fn missing_key_is_none_not_zero() {
        let service = IoService::matching("IOAccelerator").expect("IOAccelerator");
        let stats = service
            .property("PerformanceStatistics")
            .expect("PerformanceStatistics");
        assert_eq!(dict_number(&stats, "No Such Key %"), None);
    }
}
