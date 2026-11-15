//! System RAM in a browser (wasm32 / wasm64 on `target_os = "unknown"`): what a page may hold.
//!
//! A page has no OS to ask. Two facts bound it:
//! - the device RAM the browser reports, `navigator.deviceMemory` (GiB, rounded down to a power of
//!   two and capped by the browser for privacy; Chromium only), read from the global object so a
//!   window and a worker both answer;
//! - on wasm32, the address space: linear memory is at most 65536 pages of 64 KiB = 4 GiB. Memory64
//!   (wasm64) has no such bound below the device.
//!
//! `total_bytes` is the smaller of the two; `available_bytes` is that minus the linear memory the
//! module already holds (`WebAssembly.Memory.buffer.byteLength`). With no device figure, wasm32 still
//! knows its 4 GiB; wasm64 knows nothing and reports `None`, never an invented default.

use crate::os::SysMemInfo;
use wasm_bindgen::JsValue;

/// wasm32 linear memory: at most 65536 pages of 64 KiB.
const WASM32_MEMORY: u64 = 65536 * 65536;

/// `obj[key]` as a number, if it is one.
fn number(obj: &JsValue, key: &str) -> Option<f64> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key)).ok()?.as_f64()
}

/// The device RAM the browser reports, in bytes.
fn device_memory() -> Option<u64> {
    let navigator = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator")).ok()?;
    let gib = number(&navigator, "deviceMemory").filter(|g| g.is_finite() && *g > 0.0)?;
    Some((gib * f64::from(1u32 << 30)) as u64)
}

/// The linear memory this module holds now, in bytes.
fn linear_memory() -> Option<u64> {
    let buffer = js_sys::Reflect::get(&wasm_bindgen::memory(), &JsValue::from_str("buffer")).ok()?;
    number(&buffer, "byteLength").map(|b| b as u64)
}

pub(crate) fn sys_mem() -> Option<SysMemInfo> {
    let total = match (device_memory(), cfg!(target_pointer_width = "32")) {
        (Some(device), true) => device.min(WASM32_MEMORY),
        (Some(device), false) => device,
        (None, true) => WASM32_MEMORY,
        (None, false) => return None,
    };
    let held = linear_memory().unwrap_or(0).min(total);
    Some(SysMemInfo { total_bytes: total, available_bytes: total - held })
}
