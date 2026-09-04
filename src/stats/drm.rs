//! Linux GPU counters from DRM sysfs — plain file reads, no `nvidia-smi` spawn.
//!
//! `amdgpu` and `i915`/`xe` expose live counters as sysfs attributes, so a poll costs a
//! `read(2)` on a pseudo-file:
//!
//! | Attribute | Meaning |
//! |---|---|
//! | `gpu_busy_percent` | device utilisation, already in percent |
//! | `mem_info_vram_used` | VRAM bytes in use |
//! | `mem_info_vram_total` | VRAM bytes total |
//!
//! NVIDIA publishes none of these: its counters live behind NVML, and `nvidia-smi` is a
//! process spawn, which this module's contract forbids. An NVIDIA-only host therefore gets
//! `None` here and should use [`crate::os`] on its own slow path until NVML is wired up.

use super::GpuStats;

/// Card index range to probe. Eight covers every realistic multi-GPU desktop.
const MAX_CARDS: u32 = 8;

pub(super) fn query() -> Option<GpuStats> {
    (0..MAX_CARDS).find_map(|index| card(index))
}

fn card(index: u32) -> Option<GpuStats> {
    let base = format!("/sys/class/drm/card{index}/device");

    let util_pct = read_u64(&format!("{base}/gpu_busy_percent"))
        .map(|busy| (busy as f32).clamp(0.0, 100.0));
    let mem_total_bytes = read_u64(&format!("{base}/mem_info_vram_total"));
    let mem_used_bytes = read_u64(&format!("{base}/mem_info_vram_used"));

    // A card that answers none of the three is either NVIDIA or not a GPU at all.
    if util_pct.is_none() && mem_total_bytes.is_none() && mem_used_bytes.is_none() {
        return None;
    }

    Some(GpuStats {
        name: read_line(&format!("{base}/label"))
            .or_else(|| read_line(&format!("{base}/product_name"))),
        util_pct,
        mem_used_bytes,
        mem_total_bytes,
        // Discrete VRAM. An integrated part reports a tiny carve-out rather than system RAM,
        // so claiming "unified" would invite callers to size budgets off the wrong pool.
        unified: false,
        // DRM sysfs does not publish thermal, clock, power or fan here.
        ..Default::default()
    })
}

fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn read_line(path: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().next()?.trim();
    (!line.is_empty()).then(|| line.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_paths_are_none() {
        assert_eq!(read_u64("/sys/class/drm/card999/device/gpu_busy_percent"), None);
        assert_eq!(read_line("/sys/class/drm/card999/device/label"), None);
    }

    #[test]
    fn smoke_query() {
        // Hardware-dependent: only checks that probing every card is side-effect free.
        eprintln!("drm gpu stats = {:?}", query());
    }
}
