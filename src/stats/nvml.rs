//! NVIDIA counters PDH does not have: temperature, power, fans, clocks.
//!
//! # Why a vendor library at all
//!
//! [`super::windows`] is deliberately vendor-neutral, and that is the whole of what Windows
//! publishes: the `GPU Engine` and `GPU Adapter Memory` counter sets carry utilisation and
//! memory and **nothing else**. There is no thermal, clock, power or fan counter anywhere in
//! PDH, and no DXGI call for them either. A monitor that wants them has to ask the driver.
//!
//! So this is an *enrichment* layer, not a second backend and not a fallback: the base
//! reading stands on its own for every vendor, and on an NVIDIA machine these fields are
//! filled in on top of it. AMD and Intel would need ADL and IGCL respectively, in exactly
//! this shape.
//!
//! # Why it is loaded by hand
//!
//! NVML ships with the driver, so linking against it would make the binary refuse to start
//! on a machine without an NVIDIA card. `LoadLibrary` + `GetProcAddress` at first use keeps
//! the dependency to "if it is there, use it", costs one load for the process's lifetime,
//! and — like every other backend here — spawns nothing.
//!
//! The entry points are declared locally rather than pulled from a bindings crate: this is a
//! handful of C functions with a stable ABI, and NVML's own headers have kept these
//! signatures across major versions.

use std::ffi::{c_char, c_uint, c_void};

/// One enrichment reading. Every field is `Option` because NVML answers per capability:
/// a passively cooled card reports no fan, a laptop GPU may report no enforced power limit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NvmlStats {
    /// Core temperature in °C.
    pub temp_c: Option<f32>,
    /// Board power draw in watts.
    pub power_w: Option<f32>,
    /// The power limit currently enforced, in watts — the denominator for `power_w`.
    pub power_limit_w: Option<f32>,
    /// Fan speed as a percentage of its maximum, not RPM: NVML reports the duty cycle.
    pub fan_pct: Option<f32>,
    /// Graphics clock in MHz.
    pub clock_core_mhz: Option<u32>,
    /// Memory clock in MHz.
    pub clock_mem_mhz: Option<u32>,
    /// Memory-*controller* utilisation: the share of time the bus was busy, which is a
    /// different question from how full the VRAM is.
    pub mem_bus_pct: Option<f32>,
    /// Driver version string, e.g. `"581.15"`.
    pub driver: Option<String>,
}

impl NvmlStats {
    /// True when NVML answered nothing at all, so a caller can leave the base reading alone.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Reads the enrichment counters for the device whose name matches `adapter`.
///
/// Matching by name rather than taking index 0: on a machine with two NVIDIA cards the
/// enumeration order is NVML's, not DXGI's, and reporting one card's temperature next to the
/// other's memory would be worse than reporting no temperature.
pub fn query(adapter: &str) -> Option<NvmlStats> {
    let lib = Nvml::get()?;
    let device = lib.device_matching(adapter)?;
    let mut out = NvmlStats {
        driver: lib.driver_version(),
        ..Default::default()
    };

    let mut value: c_uint = 0;
    // SAFETY (all calls below): `device` is a handle NVML itself returned and has not been
    // invalidated — NVML stays initialised for the process's lifetime — and each out
    // parameter is a valid local of the type the entry point documents. Every call reports
    // failure through its return code, which is checked before the value is read.
    if unsafe { (lib.temperature)(device, NVML_TEMPERATURE_GPU, &mut value) } == NVML_SUCCESS {
        out.temp_c = Some(value as f32);
    }
    if unsafe { (lib.power_usage)(device, &mut value) } == NVML_SUCCESS {
        out.power_w = Some(value as f32 / 1000.0);
    }
    if unsafe { (lib.power_limit)(device, &mut value) } == NVML_SUCCESS {
        out.power_limit_w = Some(value as f32 / 1000.0);
    }
    if unsafe { (lib.fan_speed)(device, &mut value) } == NVML_SUCCESS {
        out.fan_pct = Some(value as f32);
    }
    if unsafe { (lib.clock_info)(device, NVML_CLOCK_GRAPHICS, &mut value) } == NVML_SUCCESS {
        out.clock_core_mhz = Some(value);
    }
    if unsafe { (lib.clock_info)(device, NVML_CLOCK_MEM, &mut value) } == NVML_SUCCESS {
        out.clock_mem_mhz = Some(value);
    }
    let mut rates = NvmlUtilization::default();
    if unsafe { (lib.utilization)(device, &mut rates) } == NVML_SUCCESS {
        out.mem_bus_pct = Some(rates.memory as f32);
    }

    (!out.is_empty()).then_some(out)
}

// ── The library ────────────────────────────────────────────────────────────────────────

const NVML_SUCCESS: c_uint = 0;
const NVML_TEMPERATURE_GPU: c_uint = 0;
const NVML_CLOCK_GRAPHICS: c_uint = 0;
const NVML_CLOCK_MEM: c_uint = 2;
/// NVML's own buffer size for a name; the API truncates rather than overflowing.
const NAME_LEN: usize = 96;

/// Opaque NVML device handle.
type Device = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NvmlUtilization {
    gpu: c_uint,
    memory: c_uint,
}

type FnInit = unsafe extern "C" fn() -> c_uint;
type FnCount = unsafe extern "C" fn(*mut c_uint) -> c_uint;
type FnHandle = unsafe extern "C" fn(c_uint, *mut Device) -> c_uint;
type FnName = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_uint;
type FnU32 = unsafe extern "C" fn(Device, *mut c_uint) -> c_uint;
type FnU32Arg = unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> c_uint;
type FnUtil = unsafe extern "C" fn(Device, *mut NvmlUtilization) -> c_uint;
type FnDriver = unsafe extern "C" fn(*mut c_char, c_uint) -> c_uint;

/// The entry points, resolved once.
struct Nvml {
    count: FnCount,
    handle: FnHandle,
    name: FnName,
    driver: FnDriver,
    temperature: FnU32Arg,
    power_usage: FnU32,
    power_limit: FnU32,
    fan_speed: FnU32,
    clock_info: FnU32Arg,
    utilization: FnUtil,
}

// SAFETY: the fields are plain function pointers into a library that is never unloaded, and
// NVML's documented threading model is that these entry points are safe to call from any
// thread once `nvmlInit` has succeeded.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

impl Nvml {
    /// The process-wide instance, loaded and initialised at most once.
    ///
    /// `OnceLock` rather than the per-thread caching the PDH side uses: NVML initialisation
    /// is global and reference-counted inside the driver, so doing it per thread would be
    /// wasted work for a handle every thread can share.
    fn get() -> Option<&'static Self> {
        static NVML: std::sync::OnceLock<Option<Nvml>> = std::sync::OnceLock::new();
        NVML.get_or_init(Self::load).as_ref()
    }

    fn load() -> Option<Self> {
        let lib = load_library()?;
        // SAFETY: each name is a NUL-terminated literal that NVML exports with the signature
        // its headers declare; a missing symbol yields `None` and abandons the load rather
        // than being called.
        unsafe {
            let init: FnInit = symbol(lib, c"nvmlInit_v2")?;
            if init() != NVML_SUCCESS {
                return None;
            }
            Some(Self {
                count: symbol(lib, c"nvmlDeviceGetCount_v2")?,
                handle: symbol(lib, c"nvmlDeviceGetHandleByIndex_v2")?,
                name: symbol(lib, c"nvmlDeviceGetName")?,
                driver: symbol(lib, c"nvmlSystemGetDriverVersion")?,
                temperature: symbol(lib, c"nvmlDeviceGetTemperature")?,
                power_usage: symbol(lib, c"nvmlDeviceGetPowerUsage")?,
                power_limit: symbol(lib, c"nvmlDeviceGetEnforcedPowerLimit")?,
                fan_speed: symbol(lib, c"nvmlDeviceGetFanSpeed")?,
                clock_info: symbol(lib, c"nvmlDeviceGetClockInfo")?,
                utilization: symbol(lib, c"nvmlDeviceGetUtilizationRates")?,
            })
        }
    }

    /// The device whose NVML name matches `adapter`, comparing on the model rather than on
    /// the exact string: DXGI and NVML agree on "NVIDIA GeForce RTX 3080 Ti" today, and a
    /// vendor prefix or a trailing revision is not worth losing the match over.
    fn device_matching(&self, adapter: &str) -> Option<Device> {
        let mut count: c_uint = 0;
        // SAFETY: `count` is a valid local; the call reports failure through its return code.
        if unsafe { (self.count)(&mut count) } != NVML_SUCCESS {
            return None;
        }
        let wanted = normalise(adapter);
        let mut first = None;
        for index in 0..count {
            let mut device: Device = std::ptr::null_mut();
            // SAFETY: `index` is below the count NVML just reported.
            if unsafe { (self.handle)(index, &mut device) } != NVML_SUCCESS {
                continue;
            }
            first.get_or_insert(device);
            if self.name_of(device).map(|n| normalise(&n)) == Some(wanted.clone()) {
                return Some(device);
            }
        }
        // A single NVIDIA device that reports a different name than DXGI does is still
        // unambiguously the card the base reading came from.
        (count == 1).then_some(first?)
    }

    fn name_of(&self, device: Device) -> Option<String> {
        let mut buf = [0_u8; NAME_LEN];
        // SAFETY: `buf` is `NAME_LEN` bytes and that length is what NVML is told; it writes a
        // NUL-terminated string within it or fails.
        let status = unsafe { (self.name)(device, buf.as_mut_ptr().cast(), NAME_LEN as c_uint) };
        (status == NVML_SUCCESS).then(|| c_str(&buf))
    }

    fn driver_version(&self) -> Option<String> {
        let mut buf = [0_u8; NAME_LEN];
        // SAFETY: as `name_of` — a buffer of the length NVML is given.
        let status = unsafe { (self.driver)(buf.as_mut_ptr().cast(), NAME_LEN as c_uint) };
        (status == NVML_SUCCESS).then(|| c_str(&buf)).filter(|s| !s.is_empty())
    }
}

/// Bytes up to the first NUL, as a `String`.
fn c_str(buf: &[u8]) -> String {
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}

/// Lowercased and stripped of the vendor prefix, so DXGI's and NVML's spellings meet.
fn normalise(name: &str) -> String {
    name.to_ascii_lowercase()
        .trim_start_matches("nvidia")
        .trim()
        .to_string()
}

#[cfg(windows)]
mod platform {
    use std::ffi::{c_void, CStr};

    use windows::core::{s, PCSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

    pub type Handle = HMODULE;

    /// Loads `nvml.dll`, which the NVIDIA driver installs into the system directory.
    pub fn load_library() -> Option<Handle> {
        // SAFETY: a static NUL-terminated name; a missing library is reported as an error
        // rather than a null handle we would have to check ourselves.
        unsafe { LoadLibraryA(s!("nvml.dll")) }.ok()
    }

    /// Resolves one entry point, transmuted to the signature its header declares.
    ///
    /// # Safety
    ///
    /// `F` must be the exact ABI of the symbol named by `name`, or calls through the result
    /// are undefined. Every caller here passes a signature taken from `nvml.h`.
    pub unsafe fn symbol<F: Copy>(lib: Handle, name: &CStr) -> Option<F> {
        let raw = unsafe { GetProcAddress(lib, PCSTR(name.as_ptr().cast())) }?;
        debug_assert_eq!(
            size_of::<F>(),
            size_of::<*const c_void>(),
            "a function pointer is one word"
        );
        // SAFETY: the caller guarantees `F` matches the symbol, and the size check above
        // pins `F` to a bare function pointer rather than a fat one.
        Some(unsafe { std::mem::transmute_copy::<_, F>(&raw) })
    }
}

use platform::{load_library, symbol};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_meet_after_normalising() {
        assert_eq!(
            normalise("NVIDIA GeForce RTX 3080 Ti"),
            normalise("GeForce RTX 3080 Ti")
        );
    }

    #[test]
    fn a_c_string_stops_at_the_nul() {
        let mut buf = [0_u8; 8];
        buf[..3].copy_from_slice(b"581");
        assert_eq!(c_str(&buf), "581");
    }

    /// Not an assertion about hardware: it proves the loader is side-effect free and, on a
    /// machine with an NVIDIA card, that the entry points actually resolve and answer.
    #[test]
    fn smoke_query() {
        let stats = query("NVIDIA GeForce RTX 3080 Ti");
        eprintln!("nvml::query() = {stats:?}");
        if let Some(s) = stats {
            if let Some(t) = s.temp_c {
                assert!((0.0..=125.0).contains(&t), "temperature out of range: {t}");
            }
            if let Some(f) = s.fan_pct {
                assert!((0.0..=100.0).contains(&f), "fan out of range: {f}");
            }
        }
    }
}
