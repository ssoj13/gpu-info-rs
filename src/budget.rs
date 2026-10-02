//! Memory budgets: THE formulas that turn "how much RAM / VRAM may this process use" into bytes.
//!
//! # Why this module exists
//! Seven projects of the cluster each invented a RAM formula (three policies, two of them silently
//! falling back to a made-up 2 GiB / 8 GiB when the OS query failed), and the VRAM share lived in
//! one of them only. The OS numbers already come from [`crate::os`]; the formulas now live beside
//! them so every consumer gets the same answer for the same machine.
//!
//! # RAM: one [`Policy`], one pure core
//! `fraction` (clamped to `0..=1`) and `reserve` (bytes the OS + GPU driver + the application keep)
//! mean the same thing in every policy; only the base RAM differs:
//!
//! | Policy | Formula | Use when |
//! |---|---|---|
//! | [`Policy::Installed`] | `min(fraction * total, total - reserve)` | a long-lived cache: a property of the MACHINE |
//! | [`Policy::Available`] | `min(fraction * avail, avail - reserve)` | a one-shot job sized to what is free now |
//! | [`Policy::Min`] (default) | `min(fraction * total, total - reserve, avail - reserve)` | both: a share of the machine, never past what is free |
//!
//! - **Installed** is the playa / scancache formula. A cache sized off FREE memory shrinks its own
//!   allowance as it fills (every cached frame is memory the OS stops reporting free) and inherits
//!   whatever else ran at launch. The reserve is a CAP, not multiplied in: multiplying both once
//!   turned a "75%" setting into 56% of installed RAM.
//! - **Available** never claims memory that is not free right now; it is deterministic only per moment.
//! - **Min** is the exv-tile formula (`fraction = 0.5`, `reserve = 0` is its `usable_ram`): a hostile
//!   file or a huge job must not push the host into swap, and a quiet machine still does not hand
//!   everything to one process. The fraction applies to the INSTALLED RAM only; free RAM is a cap.
//!
//! [`ram_budget_from`] is the pure core (unit-testable without the OS); [`ram_budget`] reads the OS. When
//! the OS cannot answer, the result is [`BudgetError::UnknownRam`], never an invented default: the
//! caller decides (an explicit `--mem`, a UI field), and [`ram_budget_or`] lets that override win
//! without querying the OS at all.
//!
//! # VRAM
//! [`vram_share`] of the headroom (0.66; a unified-memory GPU, whose "VRAM" is system RAM the CPU
//! needs too, 0.25). [`vram_budget`] reads [`crate::os::query`] (no GPU context); with a live
//! context, [`live_headroom`] reads the driver through [`crate::VramQuerier`]. [`plan_vram`] splits
//! that headroom between fixed targets, a decode pool and a resident-tile atlas.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Share of the free VRAM a resident path may plan on: the rest stays for the compositor, other
/// applications and driver growth.
pub const VRAM_SHARE: f64 = 0.66;
/// [`VRAM_SHARE`] of a unified-memory GPU (Apple Silicon, iGPU): its "VRAM" IS system RAM, which
/// the OS and any CPU decode path need too.
pub const VRAM_SHARE_UNIFIED: f64 = 0.25;
/// The decode pool of a resident GPU decoder (peak VRAM of ONE tile decode) is a `1 / POOL_DIV`
/// slice of the planned VRAM, clamped to `[POOL_MIN, POOL_MAX]`: a tile decode needs a working
/// floor, and past the cap more pool adds nothing.
pub const POOL_DIV: u64 = 16;
/// Floor of the [`plan_vram`] decode pool.
pub const POOL_MIN: u64 = 64 << 20;
/// Cap of the [`plan_vram`] decode pool.
pub const POOL_MAX: u64 = 256 << 20;

/// Which RAM figure a budget is a share of (see the module docs for the formulas and why).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Policy {
    /// `min(fraction * total, total - reserve)`: a property of the machine (long-lived caches).
    Installed,
    /// `min(fraction * avail, avail - reserve)`: what is free right now (one-shot jobs).
    Available,
    /// `min(fraction * total, total - reserve, avail - reserve)`: a share of the machine, capped by
    /// what is free.
    #[default]
    Min,
}

/// The OS could not report a memory size and no explicit override was given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BudgetError {
    /// [`crate::os::sys_mem`] failed.
    #[error("cannot determine system memory; pass an explicit memory budget")]
    UnknownRam,
    /// [`crate::os::query`] reports no GPU memory.
    #[error("cannot determine GPU memory (neither the driver nor the OS reports it)")]
    UnknownVram,
}

/// THE RAM formula, pure: `total` installed and `avail` free bytes under `policy` (module docs).
/// No floor: a caller that needs one (one frame, a minimum cache) applies its own.
#[must_use]
pub fn ram_budget_from(total: u64, avail: u64, policy: Policy, fraction: f64, reserve: u64) -> u64 {
    // `as u64` saturates and maps NaN to 0, so a hostile fraction cannot wrap.
    let share = |base: u64| {
        ((base as f64 * fraction.clamp(0.0, 1.0)) as u64).min(base.saturating_sub(reserve))
    };
    match policy {
        Policy::Installed => share(total),
        Policy::Available => share(avail),
        Policy::Min => share(total).min(avail.saturating_sub(reserve)),
    }
}

/// Installed + available physical RAM from the OS, in bytes.
///
/// # Errors
/// [`BudgetError::UnknownRam`] when the OS query fails.
pub fn ram() -> Result<(u64, u64), BudgetError> {
    let m = crate::os::sys_mem().ok_or(BudgetError::UnknownRam)?;
    Ok((m.total_bytes, m.available_bytes))
}

/// [`ram_budget_from`] over this machine's RAM.
///
/// # Errors
/// [`BudgetError::UnknownRam`] when the OS query fails (never an invented default).
pub fn ram_budget(policy: Policy, fraction: f64, reserve: u64) -> Result<u64, BudgetError> {
    let (total, avail) = ram()?;
    Ok(ram_budget_from(total, avail, policy, fraction, reserve))
}

/// The explicit `over`ride when given (the OS is not queried), else [`ram_budget`].
///
/// # Errors
/// [`BudgetError::UnknownRam`] when there is no override and the OS query fails.
pub fn ram_budget_or(
    over: Option<u64>,
    policy: Policy,
    fraction: f64,
    reserve: u64,
) -> Result<u64, BudgetError> {
    match over {
        Some(b) => Ok(b),
        None => ram_budget(policy, fraction, reserve),
    }
}

/// The share of VRAM headroom to plan on ([`VRAM_SHARE`] / [`VRAM_SHARE_UNIFIED`]).
#[must_use]
pub const fn vram_share(unified: bool) -> f64 {
    if unified {
        VRAM_SHARE_UNIFIED
    } else {
        VRAM_SHARE
    }
}

/// [`vram_share`] of `headroom` bytes, pure.
#[must_use]
pub fn vram_budget_from(headroom: u64, unified: bool) -> u64 {
    (headroom as f64 * vram_share(unified)) as u64
}

/// VRAM budget from the OS-level query (no GPU context): [`vram_budget_from`] the free VRAM, else the
/// dedicated VRAM when the platform reports no free figure.
///
/// # Errors
/// [`BudgetError::UnknownVram`] when the OS reports no GPU memory.
pub fn vram_budget() -> Result<u64, BudgetError> {
    let g = crate::os::query().ok_or(BudgetError::UnknownVram)?;
    let headroom = if g.free_vram > 0 {
        g.free_vram
    } else {
        g.dedicated_vram
    };
    if headroom == 0 {
        return Err(BudgetError::UnknownVram);
    }
    Ok(vram_budget_from(headroom, g.unified))
}

/// Live VRAM headroom of a context: the driver's budget minus what is committed (`None` when the
/// driver cannot answer now). Feeds [`plan_vram`] / [`vram_budget_from`].
#[cfg(feature = "wgpu")]
#[must_use]
pub fn live_headroom(q: &crate::VramQuerier) -> Option<u64> {
    q.query().map(|i| i.budget.saturating_sub(i.used))
}

/// What a resident GPU path may spend of VRAM, in bytes ([`plan_vram`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramPlan {
    /// Resident-tile atlas LRU budget.
    pub atlas: u64,
    /// Decode pool of one tile decode.
    pub decode: u64,
}

/// [`plan_vram`] could not plan: the GPU memory is unknown, or the fixed targets leave no room for one tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramError {
    /// Neither the driver nor the OS reports the GPU memory (never an invented default).
    Unknown,
    /// The fixed targets + decode pool exceed the plannable share of the headroom, or leave less than one tile.
    TooSmall {
        /// Targets + decode pool the path must hold regardless of the atlas.
        fixed: u64,
        /// The plannable share of the headroom ([`vram_share`]).
        planned: u64,
        /// The headroom the driver reported.
        headroom: u64,
        /// Unified-memory GPU (its "VRAM" is system RAM).
        unified: bool,
    },
}

impl std::fmt::Display for VramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => {
                f.write_str("VRAM size unknown (neither the driver nor the OS reports it)")
            }
            Self::TooSmall {
                fixed,
                planned,
                headroom,
                unified,
            } => write!(
                f,
                "needs {} MiB of VRAM for its targets + decode pool but only {} MiB of the {} MiB free may be used{}",
                fixed >> 20,
                planned >> 20,
                headroom >> 20,
                if *unified { " (unified memory)" } else { "" },
            ),
        }
    }
}

impl std::error::Error for VramError {}

/// THE VRAM plan of a resident GPU path: [`vram_share`] of the `headroom` (budget minus committed;
/// `None` = unknown), minus `fixed` bytes the path always holds (full-size targets) and the decode
/// pool (`planned / POOL_DIV` clamped to `POOL_MIN..=POOL_MAX`). The rest is the atlas LRU budget;
/// it must hold at least `one_tile` bytes.
///
/// # Errors
/// [`VramError::Unknown`] without a headroom, [`VramError::TooSmall`] when the atlas would hold less than `one_tile`.
pub fn plan_vram(
    headroom: Option<u64>,
    unified: bool,
    fixed: u64,
    one_tile: u64,
) -> Result<VramPlan, VramError> {
    let headroom = headroom.ok_or(VramError::Unknown)?;
    let planned = vram_budget_from(headroom, unified);
    let decode = (planned / POOL_DIV).clamp(POOL_MIN, POOL_MAX);
    let fixed = fixed.saturating_add(decode);
    let atlas = planned.saturating_sub(fixed);
    if atlas < one_tile {
        return Err(VramError::TooSmall {
            fixed,
            planned,
            headroom,
            unified,
        });
    }
    Ok(VramPlan { atlas, decode })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// exv-tile `usable_ram(total, avail)` = min(total / 2, avail) is `Min` at 0.5 with no reserve.
    #[test]
    fn min_half_no_reserve_is_exv_usable_ram() {
        let usable = |t, a| ram_budget_from(t, a, Policy::Min, 0.5, 0);
        assert_eq!(usable(16 * GIB, 12 * GIB), 8 * GIB);
        assert_eq!(usable(16 * GIB, 3 * GIB), 3 * GIB);
        assert_eq!(usable(0, 5), 0);
        for (t, a) in [
            (64 * GIB, 7 * GIB + 13),
            (3, 2),
            (u64::MAX, 1),
            (17 * GIB + 1, 40 * GIB),
        ] {
            assert_eq!(usable(t, a), (t / 2).min(a), "total {t} avail {a}");
        }
    }

    /// Installed = scancache `Budget::from_installed_ram` (before its `.max(1)` floor), bit for bit.
    #[test]
    fn installed_is_the_scancache_formula() {
        let scan = |installed: usize, fraction: f64, reserve_gb: f64| {
            let share = (installed as f64 * fraction.clamp(0.0, 1.0)) as usize;
            let reserve = (reserve_gb.max(0.0) * 1024.0 * 1024.0 * 1024.0) as usize;
            share.min(installed.saturating_sub(reserve))
        };
        for (inst, f, r) in [
            (64 * GIB, 0.75, 2.0),
            (16 * GIB, 0.9, 4.0),
            (GIB, 0.75, 2.0),
            (8 * GIB, 1.5, 0.5),
            (8 * GIB, -1.0, 0.0),
        ] {
            let reserve = (r * GIB as f64) as u64;
            let got = ram_budget_from(inst, 0, Policy::Installed, f, reserve);
            assert_eq!(
                got,
                scan(inst as usize, f, r) as u64,
                "installed {inst} f {f} r {r}"
            );
        }
        // "75%" means 75% of the machine; the reserve caps instead of multiplying in.
        assert_eq!(
            ram_budget_from(64 * GIB, GIB, Policy::Installed, 0.75, 2 * GIB),
            48 * GIB
        );
        assert_eq!(
            ram_budget_from(4 * GIB, GIB, Policy::Installed, 0.75, 2 * GIB),
            2 * GIB
        );
    }

    #[test]
    fn available_ignores_installed_and_min_takes_the_smaller() {
        assert_eq!(
            ram_budget_from(64 * GIB, 10 * GIB, Policy::Available, 1.0, 2 * GIB),
            8 * GIB
        );
        assert_eq!(
            ram_budget_from(64 * GIB, 10 * GIB, Policy::Available, 0.5, 2 * GIB),
            5 * GIB
        );
        // Busy machine: free RAM caps the machine share.
        assert_eq!(
            ram_budget_from(64 * GIB, 10 * GIB, Policy::Min, 0.75, 2 * GIB),
            8 * GIB
        );
        // Quiet machine: the machine share wins.
        assert_eq!(
            ram_budget_from(64 * GIB, 60 * GIB, Policy::Min, 0.75, 2 * GIB),
            48 * GIB
        );
        assert_eq!(Policy::default(), Policy::Min);
    }

    #[test]
    fn reserve_and_fraction_saturate() {
        assert_eq!(ram_budget_from(GIB, GIB, Policy::Min, 1.0, 2 * GIB), 0);
        assert_eq!(ram_budget_from(GIB, GIB, Policy::Installed, f64::NAN, 0), 0);
        assert_eq!(
            ram_budget_from(u64::MAX, u64::MAX, Policy::Installed, 1.0, 0),
            u64::MAX
        );
    }

    #[test]
    fn override_wins_and_os_query_answers() {
        assert_eq!(
            ram_budget_or(Some(2 * GIB), Policy::Min, 0.5, 0),
            Ok(2 * GIB)
        );
        let (total, avail) = ram().expect("OS memory query");
        let b = ram_budget(Policy::Min, 0.5, 0).expect("budget");
        assert!(b <= total / 2 && b <= avail);
    }

    #[test]
    fn vram_share_is_smaller_when_unified() {
        assert!(vram_share(true) < vram_share(false));
        assert_eq!(
            vram_budget_from(10 * GIB, false),
            ((10 * GIB) as f64 * VRAM_SHARE) as u64
        );
    }

    /// Same numbers as exv-tile `plan_is_a_share_of_the_headroom_after_targets_and_pool`.
    #[test]
    fn plan_is_a_share_of_the_headroom_after_targets_and_pool() {
        let fixed = 8192u64 * 8192 * 24;
        let got = plan_vram(Some(10 * GIB), false, fixed, 256 * 256 * 16).expect("fits");
        assert_eq!(got.decode, POOL_MAX);
        assert_eq!(
            got.atlas,
            ((10 * GIB) as f64 * VRAM_SHARE) as u64 - fixed - POOL_MAX
        );
        let d = plan_vram(Some(16 * GIB), false, 0, 1).expect("fits");
        let u = plan_vram(Some(16 * GIB), true, 0, 1).expect("fits");
        assert!(u.atlas < d.atlas);
    }

    /// Same numbers as exv-tile `plan_clamps_the_pool_and_refuses_unknown_or_too_small`.
    #[test]
    fn plan_clamps_the_pool_and_refuses_unknown_or_too_small() {
        let p = plan_vram(Some(GIB), false, 0, 1).expect("fits");
        assert!((POOL_MIN..=POOL_MAX).contains(&p.decode));
        assert_eq!(plan_vram(None, false, 0, 1), Err(VramError::Unknown));
        let err = plan_vram(Some(4 * GIB), false, 16384u64 * 16384 * 24, 1).unwrap_err();
        assert!(
            matches!(err, VramError::TooSmall { unified: false, .. })
                && err.to_string().contains("MiB"),
            "{err}"
        );
        let fixed = 1024u64 * 1024 * 24;
        let headroom = ((fixed + POOL_MIN + 1) as f64 / VRAM_SHARE) as u64;
        assert!(plan_vram(Some(headroom), false, fixed, 256 * 256 * 16).is_err());
    }

    /// The `TooSmall` message is exv-tile's, byte for byte (hosts surface it to the user).
    #[test]
    fn too_small_message_matches_exv_tile() {
        let e = VramError::TooSmall {
            fixed: 3 << 20,
            planned: 2 << 20,
            headroom: 4 << 20,
            unified: true,
        };
        assert_eq!(
            e.to_string(),
            "needs 3 MiB of VRAM for its targets + decode pool but only 2 MiB of the 4 MiB free may be used (unified memory)"
        );
    }
}
