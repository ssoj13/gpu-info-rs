use super::*;
use crate::ResourcePool;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct Resource {
    bytes: u64,
    drops: Arc<AtomicUsize>,
    budget: MemoryBudget,
    unlocked_drop: Arc<AtomicBool>,
}

impl Resource {
    fn new(bytes: u64, budget: &MemoryBudget) -> Self {
        Self {
            bytes,
            drops: Arc::default(),
            budget: budget.clone(),
            unlocked_drop: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl Pooled for Resource {
    type Key = u64;
    fn key(&self) -> u64 {
        self.bytes
    }
    fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        self.unlocked_drop
            .fetch_and(self.budget.0.try_lock().is_ok(), Ordering::SeqCst);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn unconfigured_and_shrunk_budgets_allow_reuse_but_no_growth() {
    let budget = MemoryBudget::default();
    assert!(matches!(
        budget.reserve(1),
        Err(AdmissionError::Unconfigured)
    ));
    assert_eq!(budget.reserve(0).expect("no new allocation").remaining(), 0);
    budget.set_limit(Some(40));
    let mut reservation = budget.reserve(40).expect("fits");
    let resource = reservation
        .charge(Resource::new(40, &budget))
        .expect("reserved");
    budget.set_limit(Some(10));
    assert!(matches!(
        budget.reserve(1),
        Err(AdmissionError::Capacity {
            used: 40,
            limit: 10,
            ..
        })
    ));
    assert!(budget.reserve(0).is_ok());
    assert_eq!(budget.snapshot().allocated, 40);
    budget.set_limit(None);
    assert!(matches!(
        budget.reserve(1),
        Err(AdmissionError::Unconfigured)
    ));
    assert!(budget.reserve(0).is_ok());
    drop(resource);
    assert_eq!(budget.snapshot().used(), 0);
}

#[test]
fn partial_construction_transfers_charge_and_abort_releases_the_remainder() {
    let budget = MemoryBudget::new(Some(100));
    let mut operation = budget.reserve(80).expect("whole operation");
    let resource = Resource::new(30, &budget);
    let drops = resource.drops.clone();
    let unlocked = resource.unlocked_drop.clone();
    let allocation = operation.charge(resource).expect("first allocation");
    assert_eq!(operation.remaining(), 50);
    assert_eq!(
        budget.snapshot(),
        MemoryUsage {
            limit: Some(100),
            reserved: 50,
            allocated: 30,
            peak: 80
        }
    );
    drop(operation);
    assert_eq!(budget.snapshot().used(), 30);
    drop(allocation);
    assert_eq!(budget.snapshot().used(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(
        unlocked.load(Ordering::SeqCst),
        "native destruction must run outside accounting lock"
    );
}

#[test]
fn wrong_descriptor_does_not_consume_the_operations_remaining_allowance() {
    let budget = MemoryBudget::new(Some(100));
    let mut operation = budget.reserve(30).expect("preflight");
    let resource = Resource::new(31, &budget);
    let drops = resource.drops.clone();
    let unlocked = resource.unlocked_drop.clone();
    assert!(matches!(
        operation.charge(resource),
        Err(AdmissionError::ReservationTooSmall {
            requested: 31,
            remaining: 30
        })
    ));
    assert_eq!(operation.remaining(), 30);
    assert_eq!(budget.snapshot().reserved, 30);
    assert_eq!(budget.snapshot().allocated, 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(unlocked.load(Ordering::SeqCst));
    drop(operation);
    assert_eq!(budget.snapshot().used(), 0);
}

#[test]
fn binding_and_reader_aliases_keep_the_old_generation_charged_across_resize() {
    let budget = MemoryBudget::new(Some(100));
    let mut operation = budget.reserve(60).expect("old frame");
    let frame = Arc::new(
        operation
            .charge(Resource::new(60, &budget))
            .expect("reserved"),
    );
    let binding_dependencies = vec![frame.clone()];
    let reader = frame.clone();
    drop(frame);
    drop(operation);
    assert_eq!(budget.snapshot().allocated, 60);
    assert!(
        budget.reserve(60).is_err(),
        "old/new overlap exceeds allowance"
    );
    drop(reader);
    assert_eq!(
        budget.snapshot().allocated,
        60,
        "binding-only owner still holds allocation"
    );
    let last_owner = binding_dependencies
        .into_iter()
        .next()
        .expect("binding dependency");
    assert_eq!(Arc::strong_count(&last_owner), 1);
    let retired = Arc::try_unwrap(last_owner).expect("unique tracked owner");
    // This test models completed/unmapped work. Uniqueness alone is not a GPU fence.
    drop(retired);
    let mut resized = budget.reserve(60).expect("old allocation released");
    let replacement = resized
        .charge(Resource::new(60, &budget))
        .expect("reserved");
    assert_eq!(budget.snapshot().peak, 60);
    drop(replacement);
    drop(resized);
    assert_eq!(budget.snapshot().used(), 0);
}

#[test]
fn idle_pool_entries_remain_charged_and_checkout_does_not_charge_twice() {
    let budget = MemoryBudget::new(Some(64));
    let pool = ResourcePool::with_limits(64, 4);
    let mut operation = budget.reserve(64).expect("two misses");
    let first = operation
        .charge(Resource::new(32, &budget))
        .expect("reserved");
    let second = operation
        .charge(Resource::new(32, &budget))
        .expect("reserved");
    let first_drops = first.resource_untracked().drops.clone();
    let second_drops = second.resource_untracked().drops.clone();
    pool.put(first);
    pool.put(second);
    drop(operation);
    assert_eq!(budget.snapshot().allocated, 64);
    assert_eq!(pool.idle_count(), (2, 64));
    budget.set_limit(None);
    let reused = pool.take(32).expect("idle compatible resource");
    let no_misses = budget.reserve(0).expect("all-idle operation");
    assert!(reused.belongs_to(&budget));
    assert!(!reused.belongs_to(&MemoryBudget::new(Some(64))));
    assert_eq!(budget.snapshot().allocated, 64);
    assert_eq!(budget.snapshot().peak, 64);
    assert_eq!(pool.idle_count(), (1, 32));
    drop(no_misses);
    pool.put(reused);
    pool.set_budget(0);
    assert_eq!(pool.idle_count(), (0, 0));
    assert_eq!(budget.snapshot().used(), 0);
    assert_eq!(first_drops.load(Ordering::SeqCst), 1);
    assert_eq!(second_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_partial_failed_operation_rolls_back_without_entering_the_pool() {
    let budget = MemoryBudget::new(Some(90));
    let pool: ResourcePool<Charged<Resource>> = ResourcePool::new(90);
    let result: Result<(), AdmissionError> = (|| {
        let mut operation = budget.reserve(90)?;
        let _temporary = operation.charge(Resource::new(40, &budget))?;
        operation.charge(Resource::new(51, &budget))?;
        Ok(())
    })();
    assert!(matches!(
        result,
        Err(AdmissionError::ReservationTooSmall {
            requested: 51,
            remaining: 50
        })
    ));
    assert_eq!(budget.snapshot().used(), 0);
    assert_eq!(budget.snapshot().peak, 90);
    assert_eq!(pool.idle_count(), (0, 0));
}

#[test]
fn a_previously_admitted_operation_survives_limit_changes() {
    let budget = MemoryBudget::new(Some(80));
    let mut operation = budget.reserve(80).expect("complete reservation");
    budget.set_limit(Some(1));
    budget.set_limit(None);
    let resource = operation
        .charge(Resource::new(60, &budget))
        .expect("already admitted");
    assert_eq!(budget.snapshot().used(), 80);
    assert_eq!(budget.snapshot().allocated, 60);
    assert_eq!(budget.snapshot().reserved, 20);
    assert!(matches!(
        budget.reserve(1),
        Err(AdmissionError::Unconfigured)
    ));
    drop(operation);
    assert_eq!(budget.snapshot().used(), 60);
    drop(resource);
    assert_eq!(budget.snapshot().used(), 0);
}

#[test]
fn a_zero_sized_resource_does_not_create_a_charge() {
    let budget = MemoryBudget::default();
    let mut operation = budget.reserve(0).expect("no growth");
    let resource = operation
        .charge(Resource::new(0, &budget))
        .expect("zero allocation");
    assert_eq!(resource.bytes(), 0);
    assert_eq!(budget.snapshot().used(), 0);
    drop(resource);
    drop(operation);
    assert_eq!(budget.snapshot().used(), 0);
}

#[test]
fn overflow_does_not_wrap_or_change_accounting() {
    let budget = MemoryBudget::new(Some(u64::MAX));
    let mut reservation = budget.reserve(u64::MAX).expect("explicit allowance");
    assert!(matches!(budget.reserve(1), Err(AdmissionError::Overflow)));
    let allocation = reservation
        .charge(Resource::new(u64::MAX, &budget))
        .expect("transfer");
    assert_eq!(budget.snapshot().allocated, u64::MAX);
    assert_eq!(budget.snapshot().reserved, 0);
    assert!(matches!(budget.reserve(1), Err(AdmissionError::Overflow)));
    drop(allocation);
    drop(reservation);
    assert_eq!(budget.snapshot().used(), 0);
    assert_eq!(budget.snapshot().peak, u64::MAX);
}

#[test]
fn concurrent_whole_operations_never_overcommit() {
    const WORKERS: usize = 16;
    let budget = MemoryBudget::new(Some(39));
    let barrier = Arc::new(Barrier::new(WORKERS + 1));
    let handles: Vec<_> = (0..WORKERS)
        .map(|_| {
            let budget = budget.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let reservation = budget.reserve(13);
                barrier.wait();
                barrier.wait();
                reservation.is_ok()
            })
        })
        .collect();
    barrier.wait();
    assert_eq!(
        budget.snapshot(),
        MemoryUsage {
            limit: Some(39),
            reserved: 39,
            allocated: 0,
            peak: 39
        }
    );
    barrier.wait();
    let admitted = handles
        .into_iter()
        .map(|handle| handle.join().expect("worker"))
        .filter(|&admitted| admitted)
        .count();
    assert_eq!(admitted, 3);
    assert_eq!(budget.snapshot().used(), 0);
}
