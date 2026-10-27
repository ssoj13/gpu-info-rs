//! Logical allocation admission, separate from observed driver headroom and idle pool limits.
//!
//! Reserve all allocation misses of an operation before creating any resources.
//! Existing live generations and idle pooled resources remain charged. Bindings,
//! views and recordings must retain the same `Arc<Charged<R>>` as their resource.

use crate::Pooled;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A coherent snapshot of this owner's logical allocation accounting.
/// Driver overhead, foreign allocations and untracked handle escapes are excluded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub limit: Option<u64>,
    pub reserved: u64,
    pub allocated: u64,
    /// Largest reserved-plus-allocated total observed by this owner.
    pub peak: u64,
}

impl MemoryUsage {
    /// Bytes already allocated or promised to complete operations.
    pub fn used(&self) -> u64 {
        self.reserved + self.allocated
    }
}

/// Shared admission for one ownership domain. No limit is invented by default.
///
/// The host configures the exact adopted context's allowance. A short metadata
/// lock makes changing that allowance and reserving bytes one atomic decision;
/// no resource construction, destruction, waiting or callbacks run under it.
#[derive(Clone, Debug, Default)]
pub struct MemoryBudget(Arc<Mutex<MemoryUsage>>);

/// An allocation operation could not reserve its complete byte requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("memory allowance has not been configured")]
    Unconfigured,
    #[error("memory byte accounting overflow")]
    Overflow,
    #[error(
        "cannot reserve {requested} bytes: {used} bytes are held against a {limit}-byte allowance"
    )]
    Capacity {
        requested: u64,
        used: u64,
        limit: u64,
    },
    #[error("allocation needs {requested} bytes but its operation reserved only {remaining}")]
    ReservationTooSmall { requested: u64, remaining: u64 },
}

impl MemoryBudget {
    pub fn new(limit: Option<u64>) -> Self {
        Self(Arc::new(Mutex::new(MemoryUsage {
            limit,
            ..MemoryUsage::default()
        })))
    }

    fn usage(&self) -> MutexGuard<'_, MemoryUsage> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Configure future growth. Shrinking or clearing the allowance never
    /// revokes outstanding reservations or frees allocations still owned.
    pub fn set_limit(&self, limit: Option<u64>) {
        self.usage().limit = limit;
    }

    pub fn snapshot(&self) -> MemoryUsage {
        *self.usage()
    }

    /// Reserve all bytes of an operation's allocation misses at once.
    ///
    /// Zero bytes always succeeds: an operation entirely served by already
    /// charged idle resources needs no new allowance, even after a shrink or
    /// while admission is disabled. Positive growth needs a configured limit.
    pub fn reserve(&self, bytes: u64) -> Result<Reservation, AdmissionError> {
        let mut usage = self.usage();
        let used = usage.used();
        let total = used.checked_add(bytes).ok_or(AdmissionError::Overflow)?;
        if bytes != 0 {
            let limit = usage.limit.ok_or(AdmissionError::Unconfigured)?;
            if total > limit {
                return Err(AdmissionError::Capacity {
                    requested: bytes,
                    used,
                    limit,
                });
            }
        }
        usage.reserved += bytes;
        usage.peak = usage.peak.max(total);
        drop(usage);
        Ok(Reservation {
            budget: self.clone(),
            remaining: bytes,
        })
    }
}

/// An operation's unallocated allowance. Drop rolls back its unused remainder.
/// This is deliberately not cloneable.
#[derive(Debug)]
#[must_use = "keep the reservation until its allocations are charged, or drop it to abort"]
pub struct Reservation {
    budget: MemoryBudget,
    remaining: u64,
}

impl Reservation {
    pub fn remaining(&self) -> u64 {
        self.remaining
    }

    /// Transfer bytes into the resource's lifetime without changing total usage.
    ///
    /// `Pooled::bytes` must describe the entire allocation, and its key must
    /// describe all properties required for reuse. For example, the standard
    /// texture key covers only single-layer, single-mip, single-sample 2D images.
    /// A rejected resource is dropped outside the accounting lock.
    pub fn charge<R: Pooled>(&mut self, resource: R) -> Result<Charged<R>, AdmissionError> {
        let bytes = resource.bytes();
        if bytes > self.remaining {
            return Err(AdmissionError::ReservationTooSmall {
                requested: bytes,
                remaining: self.remaining,
            });
        }
        let mut usage = self.budget.usage();
        usage.reserved -= bytes;
        usage.allocated += bytes;
        self.remaining -= bytes;
        drop(usage);
        Ok(Charged {
            resource,
            charge: Charge {
                budget: self.budget.clone(),
                bytes,
            },
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.usage().reserved -= self.remaining;
    }
}

#[derive(Debug)]
struct Charge {
    budget: MemoryBudget,
    bytes: u64,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget.usage().allocated -= self.bytes;
    }
}

/// One resource and its one allocation charge; share this owner through `Arc`.
///
/// Resource destruction precedes charge release. A `ResourcePool<Charged<R>>`
/// retains its charge while idle and adds no charge on checkout. Pool return
/// requires both unique tracked ownership AND successful GPU retirement/unmapping;
/// `Arc::try_unwrap` alone is not a completion proof. Failed or indeterminate
/// work must not return an allocation to the reusable pool.
///
/// Native handles cloned through `resource_untracked` can outlive this owner;
/// callers using that interoperability boundary must retain this owner themselves.
/// Such escapes cannot be included in a hard logical-memory bound.
#[derive(Debug)]
pub struct Charged<R> {
    resource: R,
    charge: Charge,
}

impl<R> Charged<R> {
    /// Borrow the native resource. Keep its charged owner through every native
    /// alias, view, binding and last submitted use.
    pub fn resource_untracked(&self) -> &R {
        &self.resource
    }

    /// Check the allocation's admission domain before returning it to an owner's pool.
    pub fn belongs_to(&self, budget: &MemoryBudget) -> bool {
        Arc::ptr_eq(&self.charge.budget.0, &budget.0)
    }
}

impl<R: Pooled> Pooled for Charged<R> {
    type Key = R::Key;

    fn key(&self) -> Self::Key {
        self.resource.key()
    }

    fn bytes(&self) -> u64 {
        self.charge.bytes
    }
}

#[cfg(test)]
mod tests;
