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

use std::ffi::{c_char, c_uint, c_ulonglong, c_void};

/// One enrichment reading. Every field is `Option` because NVML answers per capability:
/// a passively cooled card reports no fan, a laptop GPU may report no enforced power limit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NvmlStats {
    /// Device core utilisation in percent (`0.0..=100.0`), when reported by NVML.
    pub util_pct: Option<f32>,
    /// VRAM bytes in use, when reported by `nvmlDeviceGetMemoryInfo`.
    pub mem_used_bytes: Option<u64>,
    /// Total dedicated VRAM bytes, when reported by `nvmlDeviceGetMemoryInfo`.
    pub mem_total_bytes: Option<u64>,
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

/// Reads the enrichment counters for the first device whose name matches `adapter`.
#[cfg(test)]
pub fn query(adapter: &str) -> Option<NvmlStats> {
    query_nth(adapter, 0)
}

/// Reads the enrichment counters for the `ordinal`-th device whose name matches `adapter`.
///
/// On a multi-GPU node with identical NVIDIA cards (e.g. 4× RTX 4090), `ordinal` selects
/// which matching NVML handle corresponds to the `ordinal`-th DXGI adapter of that model.
#[cfg(any(windows, test))]
pub fn query_nth(adapter: &str, ordinal: usize) -> Option<NvmlStats> {
    let lib = Nvml::get()?;
    let device = lib.device_matching(adapter, ordinal)?;
    lib.sample_device(device)
}

/// Enumerates every NVIDIA GPU known to NVML along with its live counters.
///
/// Primary path on Linux NVIDIA hosts (where DRM sysfs does not publish `gpu_busy_percent`
/// or `mem_info_vram_*`) — still 100% spawn-free via `libnvidia-ml.so.1`.
#[cfg_attr(windows, allow(dead_code))]
pub fn query_all() -> Vec<(String, NvmlStats)> {
    let Some(lib) = Nvml::get() else {
        return Vec::new();
    };
    let mut count: c_uint = 0;
    if unsafe { (lib.count)(&mut count) } != NVML_SUCCESS {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut device: Device = std::ptr::null_mut();
        if unsafe { (lib.handle)(index, &mut device) } != NVML_SUCCESS {
            continue;
        }
        let name = lib.name_of(device).unwrap_or_else(|| format!("NVIDIA GPU {index}"));
        if let Some(stats) = lib.sample_device(device) {
            out.push((name, stats));
        }
    }
    out
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NvmlMemory {
    total: c_ulonglong,
    free: c_ulonglong,
    used: c_ulonglong,
}

type FnInit = unsafe extern "C" fn() -> c_uint;
type FnCount = unsafe extern "C" fn(*mut c_uint) -> c_uint;
type FnHandle = unsafe extern "C" fn(c_uint, *mut Device) -> c_uint;
type FnName = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_uint;
type FnU32 = unsafe extern "C" fn(Device, *mut c_uint) -> c_uint;
type FnU32Arg = unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> c_uint;
type FnUtil = unsafe extern "C" fn(Device, *mut NvmlUtilization) -> c_uint;
type FnMem = unsafe extern "C" fn(Device, *mut NvmlMemory) -> c_uint;
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
    power_mgmt_limit: Option<FnU32>,
    fan_speed: FnU32,
    clock_info: FnU32Arg,
    utilization: FnUtil,
    memory_info: Option<FnMem>,
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
                power_mgmt_limit: symbol(lib, c"nvmlDeviceGetPowerManagementLimit"),
                fan_speed: symbol(lib, c"nvmlDeviceGetFanSpeed")?,
                clock_info: symbol(lib, c"nvmlDeviceGetClockInfo")?,
                utilization: symbol(lib, c"nvmlDeviceGetUtilizationRates")?,
                memory_info: symbol(lib, c"nvmlDeviceGetMemoryInfo"),
            })
        }
    }

    fn sample_device(&self, device: Device) -> Option<NvmlStats> {
        let mut out = NvmlStats {
            driver: self.driver_version(),
            ..Default::default()
        };

        let mut value: c_uint = 0;
        // SAFETY (all calls below): `device` is a handle NVML itself returned and has not been
        // invalidated — NVML stays initialised for the process's lifetime — and each out
        // parameter is a valid local of the type the entry point documents. Every call reports
        // failure through its return code, which is checked before the value is read.
        if unsafe { (self.temperature)(device, NVML_TEMPERATURE_GPU, &mut value) } == NVML_SUCCESS {
            out.temp_c = Some(value as f32);
        }
        if unsafe { (self.power_usage)(device, &mut value) } == NVML_SUCCESS {
            out.power_w = Some(value as f32 / 1000.0);
        }
        if unsafe { (self.power_limit)(device, &mut value) } == NVML_SUCCESS {
            out.power_limit_w = Some(value as f32 / 1000.0);
        } else if let Some(mgmt_fn) = self.power_mgmt_limit
            && unsafe { mgmt_fn(device, &mut value) } == NVML_SUCCESS
        {
            out.power_limit_w = Some(value as f32 / 1000.0);
        }
        if unsafe { (self.fan_speed)(device, &mut value) } == NVML_SUCCESS {
            out.fan_pct = Some(value as f32);
        }
        if unsafe { (self.clock_info)(device, NVML_CLOCK_GRAPHICS, &mut value) } == NVML_SUCCESS {
            out.clock_core_mhz = Some(value);
        }
        if unsafe { (self.clock_info)(device, NVML_CLOCK_MEM, &mut value) } == NVML_SUCCESS {
            out.clock_mem_mhz = Some(value);
        }
        let mut rates = NvmlUtilization::default();
        if unsafe { (self.utilization)(device, &mut rates) } == NVML_SUCCESS {
            out.util_pct = Some((rates.gpu as f32).clamp(0.0, 100.0));
            out.mem_bus_pct = Some((rates.memory as f32).clamp(0.0, 100.0));
        }
        if let Some(mem_fn) = self.memory_info {
            let mut mem = NvmlMemory::default();
            if unsafe { mem_fn(device, &mut mem) } == NVML_SUCCESS && mem.total > 0 {
                out.mem_total_bytes = Some(mem.total);
                out.mem_used_bytes = Some(mem.used);
            }
        }

        (!out.is_empty()).then_some(out)
    }

    /// The `ordinal`-th device whose NVML name matches `adapter`, comparing on the model
    /// rather than on the exact string: DXGI and NVML agree on "NVIDIA GeForce RTX 3080 Ti"
    /// today, and a vendor prefix or a trailing revision is not worth losing the match over.
    #[cfg(any(windows, test))]
    fn device_matching(&self, adapter: &str, ordinal: usize) -> Option<Device> {
        let mut count: c_uint = 0;
        // SAFETY: `count` is a valid local; the call reports failure through its return code.
        if unsafe { (self.count)(&mut count) } != NVML_SUCCESS {
            return None;
        }
        let wanted = normalise(adapter);
        let mut first = None;
        let mut seen_matches = 0usize;
        for index in 0..count {
            let mut device: Device = std::ptr::null_mut();
            // SAFETY: `index` is below the count NVML just reported.
            if unsafe { (self.handle)(index, &mut device) } != NVML_SUCCESS {
                continue;
            }
            first.get_or_insert(device);
            if self.name_of(device).map(|n| normalise(&n)) == Some(wanted.clone()) {
                if seen_matches == ordinal {
                    return Some(device);
                }
                seen_matches += 1;
            }
        }
        // A single NVIDIA device that reports a different name than DXGI does is still
        // unambiguously the card the base reading came from.
        (count == 1 && ordinal == 0).then_some(first?)
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
        (status == NVML_SUCCESS)
            .then(|| c_str(&buf))
            .filter(|s| !s.is_empty())
    }
}

/// Bytes up to the first NUL, as a `String`.
fn c_str(buf: &[u8]) -> String {
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}

/// Lowercased and stripped of the vendor prefix, so DXGI's and NVML's spellings meet.
#[cfg(any(windows, test))]
fn normalise(name: &str) -> String {
    name.to_ascii_lowercase()
        .trim_start_matches("nvidia")
        .trim()
        .to_string()
}

#[cfg(windows)]
mod platform {
    use std::ffi::{CStr, c_void};

    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
    use windows::core::{PCSTR, s};

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

#[cfg(unix)]
mod platform {
    use std::ffi::{CStr, c_void};

    pub type Handle = *mut c_void;

    /// Loads `libnvidia-ml.so.1` (or `libnvidia-ml.so`) on Linux/Unix without linking against it.
    pub fn load_library() -> Option<Handle> {
        let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL;
        const CANDIDATES: &[&CStr] = &[
            c"libnvidia-ml.so.1",
            c"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            c"/usr/lib64/libnvidia-ml.so.1",
            c"/usr/local/nvidia/lib64/libnvidia-ml.so.1",
            c"libnvidia-ml.so",
        ];
        for cand in CANDIDATES {
            // SAFETY: static NUL-terminated library names; returns null when absent.
            let h = unsafe { libc::dlopen(cand.as_ptr(), flags) };
            if !h.is_null() {
                return Some(h);
            }
        }
        None
    }

    /// Resolves one entry point via `dlsym`.
    pub unsafe fn symbol<F: Copy>(lib: Handle, name: &CStr) -> Option<F> {
        let raw = unsafe { libc::dlsym(lib, name.as_ptr()) };
        if raw.is_null() {
            return None;
        }
        debug_assert_eq!(
            size_of::<F>(),
            size_of::<*mut c_void>(),
            "a function pointer is one word"
        );
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
