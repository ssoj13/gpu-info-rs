//! Windows system-RAM query via the `GlobalMemoryStatusEx` syscall.
//!
//! WHY this exists as its own module: [`crate::os`] is `#![forbid(unsafe_code)]` and queries every
//! platform through SAFE shells (`nvidia-smi`, `reg query`, `wmic`, sysfs, `system_profiler`). RAM on
//! Windows was read by shelling out to `wmic OS get ...` — which SPAWNS A WHOLE PROCESS (~0.3-0.5 s).
//! A consumer that polls RAM on its UI thread (e.g. rv-rs sampling its memory readout once a second)
//! therefore froze for half a second every second — a periodic playback hitch. `GlobalMemoryStatusEx`
//! is the correct modern system call for this: a MICROSECOND kernel query, no process, no I/O.
//!
//! It needs ONE `unsafe` call, so it lives here rather than in `os.rs`, isolating the unsafe exactly
//! as [`crate::vram`] already isolates the DXGI Win32 calls. The `windows` crate is already a
//! dependency (for DXGI VRAM); this only adds its `Win32_System_SystemInformation` feature.

use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

use crate::os::SysMemInfo;

/// Total + available physical RAM (bytes) via `GlobalMemoryStatusEx`, or `None` if the syscall fails.
///
/// Drop-in replacement for the old `wmic` shell in [`crate::os::sys_mem`] — same `SysMemInfo`, but a
/// syscall instead of a process spawn.
pub fn sys_mem() -> Option<SysMemInfo> {
    let mut status = MEMORYSTATUSEX {
        // The API contract: the caller MUST set `dwLength` to the struct size before the call.
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `GlobalMemoryStatusEx` writes only into `status` (a valid, correctly-sized stack local
    // whose `dwLength` is set as the API requires) and returns a `Result` we propagate. It performs no
    // I/O and spawns no process — the whole point of replacing the `wmic` shell that did.
    unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
    Some(SysMemInfo {
        total_bytes: status.ullTotalPhys,
        available_bytes: status.ullAvailPhys,
    })
}
