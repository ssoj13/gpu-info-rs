//! Windows GPU counters without a process spawn.
//!
//! # Why this is two sources rather than one
//!
//! Windows publishes no single "GPU stats" call. It splits the answer:
//!
//! * **Adapter identity and size** — DXGI. `EnumAdapters1` names the GPU and reports its
//!   `DedicatedVideoMemory`, the honest denominator for a VRAM bar, from a bare factory —
//!   no device, no swap chain, no wgpu, so a `default-features = false` consumer gets it.
//! * **Live counters** — PDH: `GPU Engine \ Utilization Percentage` and
//!   `GPU Adapter Memory \ Dedicated Usage` / `Shared Usage`. These are the numbers Task
//!   Manager draws, they
//!   are vendor-agnostic (NVIDIA, AMD, Intel alike — no NVML, no ADL), unprivileged, and
//!   reading them is an API call rather than the `nvidia-smi` spawn this module exists to
//!   avoid.
//!
//! Memory deliberately does *not* come from `IDXGIAdapter3::QueryVideoMemoryInfo`, which
//! [`crate::vram`] uses on the wgpu path. That call is scoped to the **calling process**: a
//! monitor widget asking it reports its own handful of megabytes, and a machine with a game
//! running still reads near zero. A system monitor wants what the adapter has committed
//! across every process, which is the PDH counter.
//!
//! Both handles are opened once and cached per thread, and both counters share one PDH
//! query so a poll collects them together — the module's no-spawn contract holds at UI
//! rates, at roughly a millisecond per call.
//!
//! # How utilisation is derived
//!
//! One `GPU Engine` instance exists per process *per engine* — 3D, Copy, VideoDecode and so
//! on — named like
//! `pid_9024_luid_0x00000000_0x0000A5B7_phys_0_eng_0_engtype_3D`. Summing the lot would
//! report well past 100% on a machine that is merely decoding video while it draws, because
//! separate engines run concurrently. So instances are summed *within* an engine type and
//! the busiest type wins, which is what Task Manager's headline percentage means.
//!
//! Instances are also filtered by adapter LUID. Without that a laptop's integrated GPU and
//! its discrete one would be averaged into one meaningless figure.
//!
//! # On the FFI
//!
//! DXGI and PDH have no pure-Rust route: they are the OS. All `unsafe` is confined to this
//! file behind [`query`], exactly as [`crate::win_mem`] confines `GlobalMemoryStatusEx` and
//! [`super::apple`] confines IOKit.

use std::cell::RefCell;

use windows::core::w;
use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
};
use windows::Win32::System::Performance::{
    PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
    PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY,
    PDH_MORE_DATA,
};

use super::GpuStats;

/// PDH success. The API returns a raw `u32` status rather than an `HRESULT`.
const PDH_OK: u32 = 0;

thread_local! {
    /// Cached adapter identity. `RefCell<Option<..>>` rather than `OnceCell` so a failed
    /// probe is retried on the next poll instead of poisoning the backend for good.
    static ADAPTER: RefCell<Option<Adapter>> = const { RefCell::new(None) };
    /// Cached PDH query. Thread-local because PDH handles are not `Sync`, and the sampler
    /// polls from one thread anyway.
    static COUNTERS: RefCell<Option<Counters>> = const { RefCell::new(None) };
}

/// Reads the primary GPU's live counters, or `None` when no adapter answers.
pub fn query() -> Option<GpuStats> {
    let adapter = ADAPTER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Adapter::open();
        }
        slot.clone()
    })?;

    let (util_pct, mem_used_bytes) = COUNTERS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Counters::open();
        }
        match slot.as_mut() {
            Some(c) => c.sample(&adapter.luid_tag, adapter.unified),
            None => (None, None),
        }
    });

    let stats = GpuStats {
        name: Some(adapter.name),
        util_pct,
        mem_used_bytes,
        mem_total_bytes: Some(adapter.total_bytes).filter(|b| *b > 0),
        unified: adapter.unified,
    };
    (!stats.is_empty()).then_some(stats)
}

/// Adapter facts for [`crate::os`]'s Windows path, with no process spawn:
/// `(name, total, shared, used, unified)` in bytes.
///
/// `os::query` answered this by running `nvidia-smi` and then `reg query` — two process
/// spawns, a second of wall clock, and NVIDIA-only for the useful half. Everything it wants
/// is already on the handles this module caches, so it may as well ask here first.
pub(crate) fn adapter_memory() -> Option<(String, u64, u64, u64, bool)> {
    let adapter = ADAPTER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Adapter::open();
        }
        slot.clone()
    })?;
    let used = COUNTERS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Counters::open();
        }
        slot.as_mut()
            .and_then(|c| c.sample(&adapter.luid_tag, adapter.unified).1)
    });
    Some((
        adapter.name,
        adapter.total_bytes,
        adapter.shared_bytes,
        used.unwrap_or(0),
        adapter.unified,
    ))
}

// ── DXGI: which GPU, and how big ───────────────────────────────────────────────────────

#[derive(Clone)]
struct Adapter {
    name: String,
    /// The denominator for a memory bar: dedicated VRAM on a discrete card, the system RAM
    /// the adapter may address on an integrated one.
    total_bytes: u64,
    /// System memory the adapter may address in addition to its own — DXGI's
    /// `SharedSystemMemory`. Reported for completeness; it is the *total* for a unified
    /// part, not a second pool on a discrete card.
    shared_bytes: u64,
    /// No dedicated video memory: an integrated part sharing system RAM.
    unified: bool,
    /// The `luid_0xHHHHHHHH_0xLLLLLLLL` fragment PDH puts in its instance names.
    luid_tag: String,
}

impl Adapter {
    /// Picks the hardware adapter with the most dedicated memory.
    ///
    /// Enumeration order is not preference order: index 0 is frequently the integrated GPU
    /// on a laptop that also has a discrete one. "Most VRAM" is the heuristic a renderer
    /// uses to choose a device, so the monitor reports the GPU the work runs on. The
    /// software (WARP) adapter is skipped — it is not a GPU.
    fn open() -> Option<Self> {
        // SAFETY: `CreateDXGIFactory1` takes no arguments and yields a refcounted COM
        // interface the `windows` crate releases on drop.
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
        let mut best: Option<Self> = None;
        for index in 0.. {
            // SAFETY: enumeration by index; a missing index returns an error, which ends the
            // loop rather than reading out of bounds.
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
                break;
            };
            // SAFETY: `GetDesc1` fills and returns a descriptor, reporting failure through
            // `Result`; a driver that will not describe itself is skipped.
            let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                continue;
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let dedicated = desc.DedicatedVideoMemory as u64;
            if best
                .as_ref()
                .is_some_and(|b| !b.unified && b.total_bytes >= dedicated)
            {
                continue;
            }
            // An integrated GPU has no VRAM at all, so its honest denominator is the system
            // memory it is allowed to address. Reporting `dedicated` there would be a
            // permanent zero, and a bar with a zero total renders as full.
            let unified = dedicated == 0;
            best = Some(Self {
                name: describe(&desc.Description),
                total_bytes: if unified {
                    desc.SharedSystemMemory as u64
                } else {
                    dedicated
                },
                shared_bytes: desc.SharedSystemMemory as u64,
                unified,
                luid_tag: luid_tag(desc.AdapterLuid),
            });
        }
        best
    }
}

/// `DXGI_ADAPTER_DESC1::Description` is a fixed 128-wchar buffer padded with NULs.
fn describe(raw: &[u16; 128]) -> String {
    let end = raw.iter().position(|c| *c == 0).unwrap_or(raw.len());
    String::from_utf16_lossy(&raw[..end]).trim().to_string()
}

/// The LUID as PDH spells it in an instance name: high word first, both unsigned.
fn luid_tag(luid: LUID) -> String {
    format!(
        "luid_0x{:08X}_0x{:08X}",
        luid.HighPart as u32, luid.LowPart
    )
}

// ── PDH: utilisation and memory in use ─────────────────────────────────────────────────

struct Counters {
    query: PDH_HQUERY,
    util: PDH_HCOUNTER,
    /// Dedicated (on-card) memory in use — the counter that means something on a discrete
    /// GPU, and reads a flat zero on an integrated one.
    dedicated: PDH_HCOUNTER,
    /// Shared (system) memory in use — the other half, and the only one an integrated GPU
    /// populates.
    shared: PDH_HCOUNTER,
    /// Scratch buffer for the counter arrays, reused between polls so a 1 Hz sampler does
    /// not allocate a few hundred instances' worth of memory every second.
    buffer: Vec<u8>,
}

impl Counters {
    fn open() -> Option<Self> {
        let mut query = PDH_HQUERY::default();
        // SAFETY: a null data source means "live data"; the handle is written into a valid
        // local and closed in `Drop`.
        if unsafe { PdhOpenQueryW(None, 0, &mut query) } != PDH_OK {
            return None;
        }
        // `PdhAddEnglishCounterW` rather than `PdhAddCounterW`: counter names are localised
        // per Windows UI language, and a hard-coded English path only resolves through the
        // English API.
        let util = add(query, w!(r"\GPU Engine(*)\Utilization Percentage"));
        let dedicated = add(query, w!(r"\GPU Adapter Memory(*)\Dedicated Usage"));
        let shared = add(query, w!(r"\GPU Adapter Memory(*)\Shared Usage"));
        let (Some(util), Some(dedicated), Some(shared)) = (util, dedicated, shared) else {
            // SAFETY: `query` is live and is not used again.
            unsafe { PdhCloseQuery(query) };
            return None;
        };
        let mut this = Self {
            query,
            util,
            dedicated,
            shared,
            buffer: Vec::new(),
        };
        // Utilisation is a rate, so PDH needs a previous sample to difference against. Prime
        // it here; without this the first poll after start-up reports a hard zero.
        this.collect();
        Some(this)
    }

    fn collect(&mut self) -> bool {
        // SAFETY: `self.query` is live for as long as `self`.
        unsafe { PdhCollectQueryData(self.query) == PDH_OK }
    }

    /// One reading for the adapter tagged `luid`: `(utilisation %, bytes of memory in use)`.
    ///
    /// `unified` picks which memory counter is the real one for this adapter — see the
    /// fields of [`Counters`].
    fn sample(&mut self, luid: &str, unified: bool) -> (Option<f32>, Option<u64>) {
        if !self.collect() {
            return (None, None);
        }
        let memory = if unified { self.shared } else { self.dedicated };
        (self.busiest_engine(luid), self.memory_used(memory, luid))
    }

    /// Utilisation of the busiest engine type, in percent.
    ///
    /// Engines run concurrently, so instances are summed *within* an engine type and the
    /// types compared — summing across types reports past 100% on a machine that merely
    /// decodes video while it draws. See the module docs.
    fn busiest_engine(&mut self, luid: &str) -> Option<f32> {
        let mut totals: Vec<(&'static str, f64)> = Vec::new();
        self.for_each(self.util, luid, |name, value| {
            let Some(engine) = name.split("engtype_").nth(1) else {
                return;
            };
            let engine = engine_type(engine);
            match totals.iter_mut().find(|(t, _)| *t == engine) {
                Some((_, sum)) => *sum += value,
                None => totals.push((engine, value)),
            }
        })?;
        let busiest = totals.into_iter().map(|(_, v)| v).fold(0.0_f64, f64::max);
        Some(busiest.clamp(0.0, 100.0) as f32)
    }

    /// Memory committed on this adapter, across every process.
    fn memory_used(&mut self, counter: PDH_HCOUNTER, luid: &str) -> Option<u64> {
        let mut total = 0.0_f64;
        self.for_each(counter, luid, |_, value| total += value)?;
        Some(total.max(0.0) as u64)
    }

    /// Formats one counter's instance array and hands each `(instance name, value)` whose
    /// name carries `luid` to `visit`.
    fn for_each(
        &mut self,
        counter: PDH_HCOUNTER,
        luid: &str,
        mut visit: impl FnMut(&str, f64),
    ) -> Option<()> {
        let count = self.format(counter)?;
        // SAFETY: `format` reported `count` initialised items at the head of the buffer it
        // just filled, and nothing reuses the buffer while this slice is alive.
        let items = unsafe {
            std::slice::from_raw_parts(
                self.buffer.as_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(),
                count,
            )
        };
        for item in items {
            // SAFETY: PDH wrote a NUL-terminated wide string into the buffer we own, valid
            // until the buffer is reused on the next call.
            let Ok(name) = (unsafe { item.szName.to_string() }) else {
                continue;
            };
            if !name.contains(luid) {
                continue;
            }
            // SAFETY: the array was formatted as `PDH_FMT_DOUBLE`, so `doubleValue` is the
            // live member of the union.
            let value = unsafe { item.FmtValue.Anonymous.doubleValue };
            if value.is_finite() {
                visit(&name, value);
            }
        }
        Some(())
    }

    /// Fills [`Self::buffer`] with `counter`'s instances, growing it when PDH asks for more.
    /// Returns how many items were written.
    fn format(&mut self, counter: PDH_HCOUNTER) -> Option<usize> {
        let mut size = self.buffer.len() as u32;
        let mut count = 0_u32;
        for attempt in 0..2 {
            let ptr = if self.buffer.is_empty() {
                None
            } else {
                Some(self.buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>())
            };
            // SAFETY: `size` describes `buffer` exactly; PDH either fills it or reports
            // `PDH_MORE_DATA` and writes the size it needs, which the next pass allocates.
            let status = unsafe {
                PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, ptr)
            };
            match status {
                PDH_OK if count > 0 => return Some(count as usize),
                // Nothing has run on any GPU engine since the last poll. A real zero, but
                // there is nothing to sum, so the caller keeps its previous value.
                PDH_OK => return None,
                PDH_MORE_DATA if attempt == 0 => self.buffer.resize(size as usize, 0),
                _ => return None,
            }
        }
        None
    }
}

impl Drop for Counters {
    fn drop(&mut self) {
        // SAFETY: the handle is live and closing it releases both counters with it.
        unsafe { PdhCloseQuery(self.query) };
    }
}

/// Adds one wildcard counter to `query`, or `None` when this Windows build has no such set.
fn add(query: PDH_HQUERY, path: windows::core::PCWSTR) -> Option<PDH_HCOUNTER> {
    let mut counter = PDH_HCOUNTER::default();
    // SAFETY: `path` is a static wide literal and `query` is a live handle.
    let status = unsafe { PdhAddEnglishCounterW(query, path, 0, &mut counter) };
    (status == PDH_OK).then_some(counter)
}

/// Engine names are a small fixed vocabulary, interned so grouping compares `&'static str`
/// rather than allocating a `String` per instance per poll.
fn engine_type(name: &str) -> &'static str {
    const KNOWN: &[&str] = &[
        "3D",
        "Copy",
        "Compute",
        "VideoDecode",
        "VideoEncode",
        "VideoProcessing",
        "Security",
        "Graphics_1",
    ];
    KNOWN
        .iter()
        .copied()
        .find(|k| *k == name)
        .unwrap_or("other")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_luid_is_spelled_the_way_pdh_spells_it() {
        let luid = LUID {
            LowPart: 0x0000_A5B7,
            HighPart: 0,
        };
        assert_eq!(luid_tag(luid), "luid_0x00000000_0x0000A5B7");
    }

    /// A negative `HighPart` must print as the unsigned word PDH uses, not as `-1`.
    #[test]
    fn a_negative_luid_high_word_is_printed_unsigned() {
        let luid = LUID {
            LowPart: 1,
            HighPart: -1,
        };
        assert_eq!(luid_tag(luid), "luid_0xFFFFFFFF_0x00000001");
    }

    #[test]
    fn a_description_stops_at_the_first_nul() {
        let mut raw = [0_u16; 128];
        for (slot, ch) in raw.iter_mut().zip("NVIDIA".encode_utf16()) {
            *slot = ch;
        }
        assert_eq!(describe(&raw), "NVIDIA");
    }

    /// Unknown engine types must still group together rather than being dropped, or a future
    /// Windows engine name would silently vanish from the total.
    #[test]
    fn unknown_engine_types_share_one_bucket() {
        assert_eq!(engine_type("3D"), "3D");
        assert_eq!(engine_type("SomethingNew"), "other");
    }
}
