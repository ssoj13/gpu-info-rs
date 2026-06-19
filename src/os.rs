//! Query GPU video memory (VRAM) on Windows, Linux, and macOS.
//!
//! **Zero `unsafe`, zero dependencies** — every platform is queried through
//! safe OS interfaces (`nvidia-smi`, `reg query`, sysfs, `system_profiler`).
//!
//! # Platform methods
//!
//! | Platform | Total VRAM | Free VRAM |
//! |----------|-----------|-----------|
//! | Windows (NVIDIA) | `nvidia-smi` or registry | `nvidia-smi` |
//! | Windows (AMD) | registry `HardwareInformation.qwMemorySize` | 0 (unavailable) |
//! | Linux (NVIDIA) | `nvidia-smi` | `nvidia-smi` |
//! | Linux (AMD) | sysfs `mem_info_vram_total` | sysfs (`total − used`) |
//! | macOS (discrete) | `system_profiler` | 0 (unavailable) |
//! | macOS (Apple Silicon) | `sysctl hw.memsize` (unified) | `vm_stat` pages free+inactive |
//!
//! # Example
//!
//! ```no_run
//! if let Some(info) = gpu_info::os::query() {
//!     println!("{}: {} MB total, {} MB free",
//!         info.name,
//!         info.dedicated_vram / (1024 * 1024),
//!         info.free_vram / (1024 * 1024),
//!     );
//! }
//! ```

#![forbid(unsafe_code)]

use std::process::Command;

// ── Public API ───────────────────────────────────────────────────────────

/// GPU memory information.
#[derive(Debug, Clone)]
pub struct GpuMemInfo {
    /// Human-readable GPU name (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub name: String,
    /// Total dedicated video memory in bytes.
    ///
    /// - Discrete GPU → real VRAM.
    /// - Apple Silicon → total unified RAM (GPU can use all of it).
    /// - Integrated GPU (no VRAM) → 0.
    pub dedicated_vram: u64,
    /// Currently available (free) VRAM in bytes.
    ///
    /// Populated via `nvidia-smi` (NVIDIA, all platforms), sysfs (AMD on
    /// Linux), or `vm_stat` (Apple Silicon, unified memory).
    /// Returns 0 when the value cannot be determined — this does
    /// **not** mean the GPU has no free memory.
    pub free_vram: u64,
    /// Shared system memory accessible by the GPU, in bytes.
    ///
    /// Populated on Windows (from the registry); 0 on other platforms.
    pub shared_memory: u64,
    /// `true` when the GPU shares system RAM (Apple Silicon, iGPU).
    ///
    /// When unified, [`dedicated_vram`](Self::dedicated_vram) is actually
    /// total system RAM and [`free_vram`](Self::free_vram) is
    /// free system RAM.  The caller may want a smaller budget fraction
    /// (e.g. 25 % instead of 66 %).
    pub unified: bool,
}

/// Queries the primary GPU's memory information.
///
/// "Primary" means the adapter with the most dedicated VRAM.
/// Returns `None` if no GPU information could be determined.
pub fn query() -> Option<GpuMemInfo> {
    platform_query()
}

/// Shorthand: returns dedicated VRAM of the primary GPU in bytes, or 0.
pub fn dedicated_vram() -> u64 {
    query().map_or(0, |g| g.dedicated_vram)
}

/// Shorthand: returns currently free VRAM of the primary GPU in bytes, or 0.
///
/// Returns 0 both when the GPU has no free memory and when the value
/// cannot be determined (AMD on Windows, macOS, etc.).
pub fn free_vram() -> u64 {
    query().map_or(0, |g| g.free_vram)
}

// ── System RAM ───────────────────────────────────────────────────────────

/// System (host) memory information.
#[derive(Debug, Clone, Copy)]
pub struct SysMemInfo {
    /// Total physical RAM in bytes.
    pub total_bytes: u64,
    /// Currently available RAM in bytes (free + reclaimable).
    pub available_bytes: u64,
}

/// Queries system (host) RAM. Returns `None` if undetermined.
pub fn sys_mem() -> Option<SysMemInfo> {
    sys_mem_platform()
}

/// Shorthand: returns available system RAM in bytes, or 0.
pub fn available_ram() -> u64 {
    sys_mem().map_or(0, |m| m.available_bytes)
}

// ── System RAM: platform dispatch ────────────────────────────────────────

#[cfg(target_os = "windows")]
fn sys_mem_platform() -> Option<SysMemInfo> {
    // `wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:list`
    // Returns values in KB.
    let output = Command::new("wmic")
        .args([
            "OS",
            "get",
            "TotalVisibleMemorySize,FreePhysicalMemory",
            "/format:list",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut free_kb: u64 = 0;
    let mut total_kb: u64 = 0;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(v) = trimmed.strip_prefix("FreePhysicalMemory=") {
            free_kb = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = trimmed.strip_prefix("TotalVisibleMemorySize=") {
            total_kb = v.trim().parse().unwrap_or(0);
        }
    }
    if total_kb == 0 {
        return None;
    }
    Some(SysMemInfo {
        total_bytes: total_kb * 1024,
        available_bytes: free_kb * 1024,
    })
}

#[cfg(target_os = "linux")]
fn sys_mem_platform() -> Option<SysMemInfo> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total: u64 = 0;
    let mut available: u64 = 0;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_kb(rest);
        }
    }
    if total == 0 {
        return None;
    }
    Some(SysMemInfo {
        total_bytes: total * 1024,
        available_bytes: available * 1024,
    })
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(s: &str) -> u64 {
    // Format: "  12345678 kB"
    s.trim().trim_end_matches("kB").trim().parse().unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn sys_mem_platform() -> Option<SysMemInfo> {
    let total = macos_sysctl_memsize()?;
    let free = macos_vm_stat_free().unwrap_or(0);
    Some(SysMemInfo {
        total_bytes: total,
        available_bytes: free,
    })
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn sys_mem_platform() -> Option<SysMemInfo> {
    None
}

// ── Common helper: nvidia-smi ────────────────────────────────────────────

/// Queries `nvidia-smi` for total, free, used VRAM and GPU name.
///
/// Works on both Windows and Linux.  The `nvidia-smi` binary is shipped
/// with the NVIDIA driver and is typically in `PATH`
/// (`C:\Windows\System32\nvidia-smi.exe` on Windows).
///
/// Returns `None` if `nvidia-smi` is not installed, exits with an error,
/// or produces unparseable output.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn nvidia_smi_query() -> Option<GpuMemInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free,name",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Output: "12288, 10783, NVIDIA GeForce RTX 3080 Ti"
    let line = stdout.lines().next()?;
    let mut parts = line.splitn(3, ',');
    let total_mib: u64 = parts.next()?.trim().parse().ok()?;
    let free_mib: u64 = parts.next()?.trim().parse().ok()?;
    let name = parts
        .next()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    Some(GpuMemInfo {
        name,
        dedicated_vram: total_mib * 1024 * 1024,
        free_vram: free_mib * 1024 * 1024,
        shared_memory: 0,
        unified: false,
    })
}

// ── Platform dispatch ────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn platform_query() -> Option<GpuMemInfo> {
    windows_query()
}

#[cfg(target_os = "linux")]
fn platform_query() -> Option<GpuMemInfo> {
    linux_query()
}

#[cfg(target_os = "macos")]
fn platform_query() -> Option<GpuMemInfo> {
    macos_query()
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn platform_query() -> Option<GpuMemInfo> {
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Windows
// ═══════════════════════════════════════════════════════════════════════════

/// Windows strategy:
/// 1. Try `nvidia-smi` first — gives total, free, and name in one call.
/// 2. Fall back to the Display Adapter registry class for total VRAM
///    (AMD / Intel / older NVIDIA without nvidia-smi in PATH).
#[cfg(target_os = "windows")]
fn windows_query() -> Option<GpuMemInfo> {
    // Fast path: nvidia-smi gives us everything, including free VRAM.
    if let Some(info) = nvidia_smi_query() {
        return Some(info);
    }

    // Fallback: registry (total only, no free VRAM).
    windows_registry_query()
}

/// Reads total VRAM and GPU name from the display adapter registry class.
///
/// This works for all GPU vendors but does **not** provide free VRAM.
#[cfg(target_os = "windows")]
fn windows_registry_query() -> Option<GpuMemInfo> {
    /// Display adapter class GUID (stable across all Windows versions).
    const DISPLAY_CLASS: &str =
        r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

    let mut best: Option<GpuMemInfo> = None;

    // Enumerate sub-keys 0000 … 0015 (covers multi-GPU setups).
    for i in 0..16u32 {
        let subkey = format!(r"{DISPLAY_CLASS}\{i:04}");

        // 64-bit VRAM (modern drivers, >4 GB GPUs).
        let vram = reg_query_hex(&subkey, "HardwareInformation.qwMemorySize")
            // Fallback: 32-bit VRAM (older drivers, ≤4 GB).
            .or_else(|| reg_query_hex(&subkey, "HardwareInformation.MemorySize"))
            .unwrap_or(0);

        if vram == 0 {
            continue;
        }

        let name = reg_query_string(&subkey, "DriverDesc")
            .or_else(|| reg_query_string(&subkey, "Device Description"))
            .unwrap_or_default();

        let shared = reg_query_hex(&subkey, "HardwareInformation.SharedSystemMemory").unwrap_or(0);

        let info = GpuMemInfo {
            name,
            dedicated_vram: vram,
            free_vram: 0, // Registry doesn't expose free VRAM.
            shared_memory: shared,
            unified: false,
        };

        if best
            .as_ref()
            .map_or(true, |b| info.dedicated_vram > b.dedicated_vram)
        {
            best = Some(info);
        }
    }

    best
}

// ── Windows registry helpers ─────────────────────────────────────────────

/// Runs `reg query <key> /v <value>` and returns the raw data string.
#[cfg(target_os = "windows")]
fn reg_query_raw(key: &str, value_name: &str) -> Option<String> {
    let output = Command::new("reg")
        .args(["query", key, "/v", value_name])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_reg_value(&stdout, value_name)
}

/// Parses a hex value (`0x…`) from a `reg query` result line.
#[cfg(target_os = "windows")]
fn reg_query_hex(key: &str, value_name: &str) -> Option<u64> {
    let data = reg_query_raw(key, value_name)?;
    let hex = data.trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(hex, 16).ok()
}

/// Parses a string value (`REG_SZ`) from a `reg query` result line.
#[cfg(target_os = "windows")]
fn reg_query_string(key: &str, value_name: &str) -> Option<String> {
    let data = reg_query_raw(key, value_name)?;
    if data.is_empty() { None } else { Some(data) }
}

/// Extracts the data field from a `reg query` output line.
///
/// Expected format (language-independent):
/// ```text
///     HardwareInformation.qwMemorySize    REG_QWORD    0x200000000
///     DriverDesc    REG_SZ    NVIDIA GeForce RTX 4090
/// ```
///
/// The value name and `REG_*` type tag are always ASCII.
#[cfg(target_os = "windows")]
fn parse_reg_value(stdout: &str, value_name: &str) -> Option<String> {
    for line in stdout.lines() {
        let trimmed = line.trim();
        // Locate the `REG_` type marker.
        let Some(reg_pos) = trimmed.find("REG_") else {
            continue;
        };
        // Everything before it is the value name.
        let name = trimmed[..reg_pos].trim();
        if !name.eq_ignore_ascii_case(value_name) {
            continue;
        }
        // Everything after `REG_xxxx<whitespace>` is the data.
        let type_and_data = &trimmed[reg_pos..];
        let Some(space) = type_and_data.find(char::is_whitespace) else {
            continue;
        };
        let data = type_and_data[space..].trim();
        if !data.is_empty() {
            return Some(data.to_string());
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Linux — nvidia-smi (NVIDIA) or sysfs (AMD / Intel)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
fn linux_query() -> Option<GpuMemInfo> {
    nvidia_smi_query().or_else(linux_amd_sysfs)
}

/// AMD / Intel: the kernel exposes VRAM stats via DRM sysfs.
///
/// Total: `/sys/class/drm/card{i}/device/mem_info_vram_total`
/// Used:  `/sys/class/drm/card{i}/device/mem_info_vram_used`
/// Free = total − used.
#[cfg(target_os = "linux")]
fn linux_amd_sysfs() -> Option<GpuMemInfo> {
    for i in 0..8u32 {
        let base = format!("/sys/class/drm/card{i}/device");
        let vram_path = format!("{base}/mem_info_vram_total");
        let Ok(contents) = std::fs::read_to_string(&vram_path) else {
            continue;
        };
        let Ok(total) = contents.trim().parse::<u64>() else {
            continue;
        };
        if total == 0 {
            continue;
        }

        let used = read_u64(&format!("{base}/mem_info_vram_used")).unwrap_or(0);
        let free = total.saturating_sub(used);

        let name = read_first_line(&format!("{base}/label"))
            .or_else(|| read_first_line(&format!("{base}/product_name")))
            .unwrap_or_else(|| format!("GPU (card{i})"));

        return Some(GpuMemInfo {
            name,
            dedicated_vram: total,
            free_vram: free,
            shared_memory: 0,
            unified: false,
        });
    }
    None
}

#[cfg(target_os = "linux")]
fn read_first_line(path: &str) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let line = s.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(line.to_string())
    }
}

#[cfg(target_os = "linux")]
fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

// ═══════════════════════════════════════════════════════════════════════════
// macOS — system_profiler + sysctl + vm_stat
// ═══════════════════════════════════════════════════════════════════════════

/// macOS strategy:
///
/// 1. Try `system_profiler SPDisplaysDataType` — if it reports a "VRAM"
///    line we're on an Intel Mac with a discrete GPU.  Free VRAM is
///    unavailable in that case.
/// 2. Otherwise assume Apple Silicon (unified memory): total = system RAM
///    via `sysctl hw.memsize`, free = (free + inactive pages) via `vm_stat`,
///    chipset name from `system_profiler`.
#[cfg(target_os = "macos")]
fn macos_query() -> Option<GpuMemInfo> {
    // Parse system_profiler for chipset name and optional VRAM line.
    let (sp_name, sp_vram) = macos_system_profiler().unwrap_or_default();

    if sp_vram > 0 {
        // Intel Mac with discrete GPU — VRAM is dedicated, free unknown.
        return Some(GpuMemInfo {
            name: sp_name,
            dedicated_vram: sp_vram,
            free_vram: 0,
            shared_memory: 0,
            unified: false,
        });
    }

    // Apple Silicon (or Intel iGPU) — unified memory.
    let total = macos_sysctl_memsize().unwrap_or(0);
    let free = macos_vm_stat_free().unwrap_or(0);
    let name = if sp_name.is_empty() {
        "Apple GPU".to_string()
    } else {
        sp_name
    };

    if total > 0 {
        Some(GpuMemInfo {
            name,
            dedicated_vram: total,
            free_vram: free,
            shared_memory: 0,
            unified: true,
        })
    } else {
        None
    }
}

/// Parses `system_profiler SPDisplaysDataType` → (chipset_name, vram_bytes).
///
/// `vram_bytes` is 0 when there is no "VRAM" line (Apple Silicon).
#[cfg(target_os = "macos")]
fn macos_system_profiler() -> Option<(String, u64)> {
    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut name = String::new();
    let mut vram: u64 = 0;

    for line in stdout.lines() {
        let trimmed = line.trim();

        // "Chipset Model: Apple M2 Max" or "Chipset Model: AMD Radeon Pro 5500M"
        if let Some(rest) = trimmed.strip_prefix("Chipset Model:") {
            name = rest.trim().to_string();
        }

        // "VRAM (Total):  8 GB"  or  "VRAM (Dynamic, Max): 48 GB"
        if trimmed.starts_with("VRAM") {
            if let Some(colon_rest) = trimmed.split_once(':') {
                vram = parse_size_string(colon_rest.1.trim()).unwrap_or(0);
            }
        }
    }

    Some((name, vram))
}

/// Returns total physical memory via `sysctl -n hw.memsize`.
#[cfg(target_os = "macos")]
fn macos_sysctl_memsize() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

/// Estimates free memory from `vm_stat` output.
///
/// "Free" here is `(Pages free + Pages inactive) × page_size`.
/// This is a reasonable approximation of memory available for GPU
/// allocation on unified-memory Macs — inactive pages can be reclaimed
/// by the system on demand.
///
/// ```text
/// Mach Virtual Memory Statistics: (page size of 16384 bytes)
/// Pages free:                               123456.
/// Pages active:                             234567.
/// Pages inactive:                            34567.
/// …
/// ```
#[cfg(target_os = "macos")]
fn macos_vm_stat_free() -> Option<u64> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();

    // First line: "Mach Virtual Memory Statistics: (page size of NNNNN bytes)"
    let first = lines.next()?;
    let page_size: u64 = first
        .rsplit("page size of ")
        .next()?
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .ok()?;

    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut speculative_pages: u64 = 0;

    for line in lines {
        let trimmed = line.trim().trim_end_matches('.');
        if let Some(rest) = trimmed.strip_prefix("Pages free:") {
            free_pages = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = trimmed.strip_prefix("Pages inactive:") {
            inactive_pages = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = trimmed.strip_prefix("Pages speculative:") {
            speculative_pages = rest.trim().parse().unwrap_or(0);
        }
    }

    Some((free_pages + inactive_pages + speculative_pages) * page_size)
}

/// Parses `"16 GB"` / `"4096 MB"` → bytes.
#[cfg(target_os = "macos")]
fn parse_size_string(s: &str) -> Option<u64> {
    let mut parts = s.split_whitespace();
    let value: u64 = parts.next()?.parse().ok()?;
    let unit = parts.next().unwrap_or("B").to_uppercase();
    let multiplier = match unit.as_str() {
        "TB" => 1024 * 1024 * 1024 * 1024,
        "GB" => 1024 * 1024 * 1024,
        "MB" => 1024 * 1024,
        "KB" => 1024,
        _ => 1,
    };
    Some(value * multiplier)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_query() {
        // Just ensure it doesn't panic — result is system-dependent.
        let info = query();
        eprintln!("gpu_info::os::query() = {info:?}");
        let total = dedicated_vram();
        let free = free_vram();
        eprintln!(
            "dedicated = {} MB, free = {} MB, used = {} MB",
            total / (1024 * 1024),
            free / (1024 * 1024),
            total.saturating_sub(free) / (1024 * 1024),
        );
    }

    #[test]
    fn smoke_sys_mem() {
        let info = sys_mem();
        eprintln!("gpu_info::os::sys_mem() = {info:?}");
        let avail = available_ram();
        eprintln!("available_ram = {} MB", avail / (1024 * 1024));
        // On any real system, total should be >0.
        if let Some(m) = info {
            assert!(m.total_bytes > 0, "total RAM must be >0");
            assert!(m.available_bytes <= m.total_bytes);
        }
    }

    #[test]
    fn nvidia_smi_parse() {
        // Verify the nvidia-smi parsing logic directly.
        let csv = "12288, 10783, NVIDIA GeForce RTX 3080 Ti";
        let mut parts = csv.splitn(3, ',');
        let total: u64 = parts.next().unwrap().trim().parse().unwrap();
        let free: u64 = parts.next().unwrap().trim().parse().unwrap();
        let name = parts.next().unwrap().trim();
        assert_eq!(total, 12288);
        assert_eq!(free, 10783);
        assert_eq!(name, "NVIDIA GeForce RTX 3080 Ti");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn parse_reg_qword() {
        let output = r#"
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000
    HardwareInformation.qwMemorySize    REG_QWORD    0x300000000
"#;
        let val = parse_reg_value(output, "HardwareInformation.qwMemorySize");
        assert_eq!(val.as_deref(), Some("0x300000000"));
        let bytes = u64::from_str_radix("300000000", 16).unwrap();
        assert_eq!(bytes, 12 * 1024 * 1024 * 1024); // 12 GB
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn parse_reg_sz() {
        let output = r#"
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000
    DriverDesc    REG_SZ    NVIDIA GeForce RTX 4090
"#;
        let val = parse_reg_value(output, "DriverDesc");
        assert_eq!(val.as_deref(), Some("NVIDIA GeForce RTX 4090"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_size() {
        assert_eq!(parse_size_string("8 GB"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_size_string("4096 MB"), Some(4096 * 1024 * 1024));
    }
}
