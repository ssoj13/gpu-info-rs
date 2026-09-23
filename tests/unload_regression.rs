//! Unload regression for [`gpu_info::shared_device`]: a module that negotiated the shared device
//! and is then unloaded must not leave threads running its code, nor leak a device per reload.
//!
//! Builds the `unload_probe_dll` fixture and the `unload_probe` probe (see their docs) into a
//! separate target directory, so this nested `cargo build` never waits on the lock of the build
//! that runs the test, then runs the probe and requires exit code 0. Windows only (the probe
//! enumerates threads by start address); ignored because it needs a GPU and builds wgpu.
#![cfg(windows)]

use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "requires a GPU; builds the probe examples (minutes)"]
fn unloaded_module_leaves_no_thread_and_leaks_no_device() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.join("target").join("unload-probe");
    let build = Command::new(env!("CARGO"))
        .current_dir(&manifest_dir)
        .args([
            "build",
            "--example",
            "unload_probe_dll",
            "--example",
            "unload_probe",
        ])
        .arg("--target-dir")
        .arg(&target_dir)
        .status()
        .expect("run cargo build for the probe examples");
    assert!(
        build.success(),
        "building the probe examples failed: {build}"
    );
    let probe = target_dir
        .join("debug")
        .join("examples")
        .join("unload_probe.exe");
    let output = Command::new(&probe)
        .output()
        .unwrap_or_else(|error| panic!("run {}: {error}", probe.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{stdout}");
    assert_eq!(
        output.status.code(),
        Some(0),
        "unload probe did not pass:\n{stdout}"
    );
}
