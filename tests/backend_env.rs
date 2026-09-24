//! `WGPU_BACKEND` selects [`gpu_info::shared_device`]'s backend for the process.
//!
//! The shared device is negotiated once per process, so each case runs this test binary again as a
//! child with the variable set, running only `child_reports_the_shared_backend`, which prints the
//! backend it got. A backend with no adapter on this machine is reported as SKIPPED with its
//! reason, never passed silently. Ignored: needs a GPU.
#![cfg(feature = "wgpu")]

use gpu_info::wgpu;
use std::process::Command;

/// Marker the child prints before its backend.
const MARKER: &str = "SHARED_BACKEND=";

/// Run this test binary as a child with `WGPU_BACKEND` = `backend` and return the shared device's
/// backend it reports.
fn shared_backend_under(backend: &str) -> String {
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args([
            "--exact",
            "child_reports_the_shared_backend",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("WGPU_BACKEND", backend)
        .env("GPU_INFO_BACKEND_CHILD", "1")
        .output()
        .expect("run the child test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "child failed under WGPU_BACKEND={backend}:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
        .lines()
        .find_map(|line| line.split_once(MARKER).map(|(_, backend)| backend))
        .unwrap_or_else(|| panic!("child printed no {MARKER} line:\n{stdout}"))
        .trim()
        .to_string()
}

#[test]
#[ignore = "requires a GPU; runs this binary as children with WGPU_BACKEND set"]
fn wgpu_backend_selects_the_shared_device_backend() {
    let mut checked = 0;
    for (value, backends, expected) in [
        ("dx12", wgpu::Backends::DX12, "Dx12"),
        ("vulkan", wgpu::Backends::VULKAN, "Vulkan"),
    ] {
        if gpu_info::query_backends(backends).adapters.is_empty() {
            println!("SKIPPED: WGPU_BACKEND={value}: no {expected} adapter on this machine");
            continue;
        }
        let got = shared_backend_under(value);
        println!("WGPU_BACKEND={value} -> shared device backend {got}");
        assert_eq!(got, expected, "WGPU_BACKEND={value}");
        checked += 1;
    }
    assert!(
        checked > 0,
        "no backend could be checked: no DX12 or Vulkan adapter"
    );
}

/// The child half of `wgpu_backend_selects_the_shared_device_backend`. Run on its own (for
/// example by `cargo test -- --ignored`) it checks nothing and says so.
#[test]
#[ignore = "helper: run by wgpu_backend_selects_the_shared_device_backend"]
fn child_reports_the_shared_backend() {
    if std::env::var_os("GPU_INFO_BACKEND_CHILD").is_none() {
        println!("helper only: run as the child of wgpu_backend_selects_the_shared_device_backend");
        return;
    }
    let gpu = gpu_info::shared_device().expect("a shared device for the selected backend");
    println!("{MARKER}{:?}", gpu.adapter.get_info().backend);
}
