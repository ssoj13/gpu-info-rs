//! Unload regression probe (Windows): load the `unload_probe_dll` fixture, negotiate the shared
//! device through it and `FreeLibrary` it, [`CYCLES`] times, and check what each cycle leaves
//! behind. Exit code 0 = pass, 1 = regression, 2 = cannot run (no GPU, fixture missing).
//!
//! Run through `cargo test --test unload_regression -- --ignored`, or by hand:
//! `cargo build --examples && target/debug/examples/unload_probe` (the fixture is found next to
//! this executable).
//!
//! Pass criteria (every one is a measured property, not a timing):
//! - the fixture image stays mapped after `FreeLibrary` (the shared device pinned it), so every
//!   load reuses the same image and the negotiation happens once;
//! - exactly ONE live thread starts inside the image in every cycle (wgpu's GL-backend instance
//!   thread, running code that stays mapped). Without the pin this count grows by one per cycle,
//!   and each such thread executes unmapped code when it next wakes: the access violation;
//! - private memory and the handle count stay flat from cycle 1 to the last cycle (without the pin
//!   each cycle leaked a whole device, measured at about 230 MB and 57 handles).
#[cfg(windows)]
mod probe {
    use std::ffi::c_void;

    /// Load/negotiate/unload cycles.
    pub const CYCLES: usize = 6;
    /// Largest private-memory growth from cycle 1 to the last cycle accepted as flat (one leaked
    /// device is ~230 MB).
    const PRIVATE_GROWTH_LIMIT: usize = 32 << 20;
    /// Largest handle-count growth accepted as flat (one leaked device is ~57 handles).
    const HANDLE_GROWTH_LIMIT: u32 = 8;

    #[repr(C)]
    #[derive(Default)]
    struct MemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged_pool: usize,
        quota_paged_pool: usize,
        quota_peak_non_paged_pool: usize,
        quota_non_paged_pool: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ModuleInfo {
        base: usize,
        size: u32,
        entry: usize,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ThreadEntry32 {
        size: u32,
        usage: u32,
        thread_id: u32,
        owner_process_id: u32,
        base_priority: i32,
        delta_priority: i32,
        flags: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn GetProcessHandleCount(process: *mut c_void, count: *mut u32) -> i32;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut MemoryCounters,
            cb: u32,
        ) -> i32;
        fn K32GetModuleInformation(
            process: *mut c_void,
            module: *mut c_void,
            info: *mut ModuleInfo,
            cb: u32,
        ) -> i32;
        fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> *mut c_void;
        fn Thread32First(snapshot: *mut c_void, entry: *mut ThreadEntry32) -> i32;
        fn Thread32Next(snapshot: *mut c_void, entry: *mut ThreadEntry32) -> i32;
        fn OpenThread(access: u32, inherit: i32, thread_id: u32) -> *mut c_void;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQueryInformationThread(
            thread: *mut c_void,
            class: u32,
            info: *mut c_void,
            length: u32,
            returned: *mut u32,
        ) -> i32;
    }

    /// `TH32CS_SNAPTHREAD`.
    const SNAP_THREAD: u32 = 0x4;
    /// `THREAD_QUERY_INFORMATION`.
    const THREAD_QUERY_INFORMATION: u32 = 0x0040;
    /// `ThreadQuerySetWin32StartAddress`.
    const THREAD_START_ADDRESS: u32 = 9;
    /// `INVALID_HANDLE_VALUE`.
    const INVALID_HANDLE: *mut c_void = usize::MAX as *mut c_void;

    /// What one cycle left behind.
    struct Cycle {
        base: usize,
        still_mapped: bool,
        threads_in_image: usize,
        handles: u32,
        private_bytes: usize,
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    /// Live threads of this process whose Win32 start address lies in `[low, high)`.
    fn threads_starting_in(low: usize, high: usize) -> Result<usize, String> {
        // SAFETY: plain Win32 calls on handles this function opens and closes itself; every
        // out-parameter points at a live local of the documented type and size.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(SNAP_THREAD, 0);
            if snapshot == INVALID_HANDLE {
                return Err("CreateToolhelp32Snapshot failed".into());
            }
            let mut entry = ThreadEntry32 {
                size: size_of::<ThreadEntry32>() as u32,
                ..Default::default()
            };
            let mut count = 0;
            let mut more = Thread32First(snapshot, &mut entry);
            while more != 0 {
                if entry.owner_process_id == GetCurrentProcessId() {
                    let thread = OpenThread(THREAD_QUERY_INFORMATION, 0, entry.thread_id);
                    if !thread.is_null() {
                        let mut start = 0usize;
                        let status = NtQueryInformationThread(
                            thread,
                            THREAD_START_ADDRESS,
                            (&raw mut start).cast(),
                            size_of::<usize>() as u32,
                            std::ptr::null_mut(),
                        );
                        if status >= 0 && (low..high).contains(&start) {
                            count += 1;
                        }
                        CloseHandle(thread);
                    }
                }
                more = Thread32Next(snapshot, &mut entry);
            }
            CloseHandle(snapshot);
            Ok(count)
        }
    }

    /// One load / negotiate / unload cycle of the fixture at `path`.
    fn cycle(path: &[u16]) -> Result<Option<Cycle>, String> {
        // SAFETY: `path` is NUL-terminated; the module handle is used only while loaded (the
        // `K32GetModuleInformation` after `FreeLibrary` only asks whether it is still mapped);
        // `negotiate` has the fixture's exact `extern "C" fn() -> i32` signature.
        unsafe {
            let module = LoadLibraryW(path.as_ptr());
            if module.is_null() {
                return Err("LoadLibraryW failed (build the examples first)".into());
            }
            let process = GetCurrentProcess();
            let mut info = ModuleInfo::default();
            if K32GetModuleInformation(process, module, &mut info, size_of::<ModuleInfo>() as u32)
                == 0
            {
                return Err("K32GetModuleInformation failed".into());
            }
            let symbol = GetProcAddress(module, c"negotiate".as_ptr().cast());
            if symbol.is_null() {
                return Err("fixture has no `negotiate` export".into());
            }
            let negotiate: extern "C" fn() -> i32 = std::mem::transmute(symbol);
            let available = negotiate() == 1;
            FreeLibrary(module);
            if !available {
                return Ok(None);
            }
            let mut probe = ModuleInfo::default();
            let still_mapped = K32GetModuleInformation(
                process,
                module,
                &mut probe,
                size_of::<ModuleInfo>() as u32,
            ) != 0;
            let mut handles = 0u32;
            GetProcessHandleCount(process, &mut handles);
            let mut counters = MemoryCounters {
                cb: size_of::<MemoryCounters>() as u32,
                ..Default::default()
            };
            K32GetProcessMemoryInfo(process, &mut counters, counters.cb);
            Ok(Some(Cycle {
                base: info.base,
                still_mapped,
                threads_in_image: threads_starting_in(info.base, info.base + info.size as usize)?,
                handles,
                private_bytes: counters.private_usage,
            }))
        }
    }

    /// Run the cycles; `Ok(true)` pass, `Ok(false)` regression, `Err` cannot run.
    pub fn run() -> Result<bool, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let fixture = exe.with_file_name("unload_probe_dll.dll");
        if !fixture.is_file() {
            return Err(format!("fixture {} not found", fixture.display()));
        }
        let path = wide(&fixture.to_string_lossy());
        let mut cycles = Vec::with_capacity(CYCLES);
        for index in 1..=CYCLES {
            let Some(cycle) = cycle(&path)? else {
                return Err("no shared GPU device on this machine".into());
            };
            println!(
                "cycle {index}: base={:#x} still_mapped={} threads_in_image={} handles={} private_mb={}",
                cycle.base,
                cycle.still_mapped,
                cycle.threads_in_image,
                cycle.handles,
                cycle.private_bytes >> 20
            );
            cycles.push(cycle);
        }
        let mut pass = true;
        for (index, cycle) in cycles.iter().enumerate() {
            if !cycle.still_mapped {
                println!(
                    "FAIL cycle {}: the image was unmapped by FreeLibrary",
                    index + 1
                );
                pass = false;
            }
            if cycle.threads_in_image != 1 {
                println!(
                    "FAIL cycle {}: {} live threads start inside the image (expected 1)",
                    index + 1,
                    cycle.threads_in_image
                );
                pass = false;
            }
        }
        if let (Some(first), Some(last)) = (cycles.first(), cycles.last()) {
            let memory = last.private_bytes.saturating_sub(first.private_bytes);
            let handles = last.handles.saturating_sub(first.handles);
            if memory > PRIVATE_GROWTH_LIMIT {
                println!(
                    "FAIL: private memory grew {} MB over the cycles",
                    memory >> 20
                );
                pass = false;
            }
            if handles > HANDLE_GROWTH_LIMIT {
                println!("FAIL: the handle count grew by {handles} over the cycles");
                pass = false;
            }
        }
        Ok(pass)
    }
}

#[cfg(windows)]
fn main() {
    match probe::run() {
        Ok(true) => println!("PASS"),
        Ok(false) => std::process::exit(1),
        Err(reason) => {
            println!("CANNOT RUN: {reason}");
            std::process::exit(2);
        }
    }
}

#[cfg(not(windows))]
fn main() {
    println!("CANNOT RUN: the unload probe enumerates threads with Win32 APIs (Windows only)");
    std::process::exit(2);
}
