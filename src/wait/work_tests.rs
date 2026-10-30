use super::*;
use crate::wait::{Notify, Signal, wait, wait_idle};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    Submitted(u32),
    Completed(u32, bool),
    Aborted(u32),
    Dropped(u32),
}

struct Lease {
    id: u32,
    events: Arc<Mutex<Vec<Event>>>,
    allocation: Option<Arc<Allocation>>,
}

impl WorkLease for Lease {
    fn submitted(&mut self) {
        self.events.lock().unwrap().push(Event::Submitted(self.id));
    }
    fn completed(self: Box<Self>, outcome: Result<(), WaitError>) {
        self.events
            .lock()
            .unwrap()
            .push(Event::Completed(self.id, outcome.is_ok()));
    }
    fn aborted(self: Box<Self>) {
        self.events.lock().unwrap().push(Event::Aborted(self.id));
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.events.lock().unwrap().push(Event::Dropped(self.id));
        // Keep the allocation observable until the whole lease, not just its hook, drops.
        let _ = self.allocation.as_ref();
    }
}

struct Allocation(Arc<AtomicUsize>);
impl Drop for Allocation {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn lease(id: u32, events: &Arc<Mutex<Vec<Event>>>) -> Lease {
    Lease {
        id,
        events: Arc::clone(events),
        allocation: None,
    }
}

fn record(context: &WorkContext, reservation: Lease) -> RecordedWork {
    let encoder = context.device().create_command_encoder(&Default::default());
    RecordedWork::new(context, [encoder.finish()], reservation)
}

fn owner(context: &WorkContext) -> WorkQueue {
    WorkQueue::new(context, NonZeroUsize::new(1).unwrap())
}

#[test]
fn completion_status_is_repeatable_and_preserves_failure() {
    let observer = Submission::pending();
    assert!(observer.status().is_none());
    let mut notify = Notify {
        signal: Arc::clone(&observer.signal),
        timed: None,
    };
    notify.finish(Err(WaitError::Poll(wgpu::PollError::WrongSubmissionIndex(
        4, 3,
    ))));
    for _ in 0..2 {
        assert!(matches!(
            observer.status(),
            Some(Err(WaitError::Poll(wgpu::PollError::WrongSubmissionIndex(
                4, 3
            ))))
        ));
    }
    drop(notify);
    assert!(matches!(observer.status(), Some(Err(WaitError::Poll(_)))));
}

#[test]
#[ignore = "requires GPU"]
fn abort_never_commits_and_settles_once() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    drop(record(&context, lease(1, &events)));
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Aborted(1), Event::Dropped(1)]
    );
}

#[test]
#[ignore = "requires GPU"]
fn completion_drains_only_in_owner_after_submission_handle_drop() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut queue = owner(&context);
    let submission = queue
        .try_submit("owned lifetime", record(&context, lease(1, &events)))
        .unwrap();
    assert_eq!(*events.lock().unwrap(), [Event::Submitted(1)]);
    assert_eq!(queue.pending(), 1);
    drop(submission);
    wait_idle(&gpu.device, &gpu.queue).unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Submitted(1)],
        "the wgpu completion callback must not destroy renderer leases"
    );
    assert_eq!(queue.collect().unwrap(), 1);
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Submitted(1),
            Event::Completed(1, true),
            Event::Dropped(1)
        ]
    );
}

#[test]
#[ignore = "requires GPU"]
fn composition_preserves_hooks_and_shared_allocation_lifetime() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let destroyed = Arc::new(AtomicUsize::new(0));
    let allocation = Arc::new(Allocation(Arc::clone(&destroyed)));
    let mut first = lease(1, &events);
    first.allocation = Some(Arc::clone(&allocation));
    let mut second = lease(2, &events);
    second.allocation = Some(Arc::clone(&allocation));
    let mut work = record(&context, first);
    work.append(record(&context, second)).unwrap();
    drop(allocation);
    let mut queue = owner(&context);
    let submitted = queue.try_submit("owned composition", work).unwrap();
    assert_eq!(destroyed.load(Ordering::SeqCst), 0);
    wait(&gpu.device, &submitted).unwrap();
    wait(&gpu.device, &submitted).unwrap();
    assert!(matches!(submitted.status(), Some(Ok(()))));
    assert_eq!(destroyed.load(Ordering::SeqCst), 0);
    queue.collect().unwrap();
    assert_eq!(destroyed.load(Ordering::SeqCst), 1);
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Submitted(1),
            Event::Submitted(2),
            Event::Completed(1, true),
            Event::Dropped(1),
            Event::Completed(2, true),
            Event::Dropped(2)
        ]
    );
}

#[test]
#[ignore = "requires GPU"]
fn capacity_rejection_returns_the_whole_unsubmitted_work() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut queue = owner(&context);
    // A controlled callback seam keeps one pending reservation unresolved.
    let signal: Arc<Signal> = Arc::default();
    queue.pending.push(PendingWork {
        completion: Submission {
            signal: Arc::clone(&signal),
            validation: None,
        },
        leases: vec![Box::new(lease(1, &events))],
        failure: None,
        recovery: None,
        committed: true,
    });
    let rejected = queue.try_submit("owned capacity", record(&context, lease(2, &events)));
    let Err(WorkSubmitError::Busy(work)) = rejected else {
        panic!("must return busy")
    };
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(queue.pending(), 1);
    drop(work);
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Aborted(2), Event::Dropped(2)]
    );
    let mut notify = Notify {
        signal,
        timed: None,
    };
    notify.finish(Ok(()));
    queue.collect().unwrap();
    assert_eq!(queue.pending(), 0);
}

#[test]
#[ignore = "requires GPU"]
fn foreign_queue_is_rejected_without_detaching_commands_or_leases() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let foreign = crate::compute_device().expect("compute GPU");
    let foreign_context = WorkContext::new(&foreign.device, &foreign.queue);
    eprintln!(
        "raw queue equality across instances: {}",
        gpu.queue == foreign.queue
    );
    assert_ne!(context, foreign_context);
    assert_eq!(context, context.clone());
    assert_ne!(context, WorkContext::new(&gpu.device, &gpu.queue));
    let events = Arc::new(Mutex::new(Vec::new()));
    let encoder = foreign.device.create_command_encoder(&Default::default());
    let work = RecordedWork::new(&foreign_context, [encoder.finish()], lease(1, &events));
    let mut own = record(&context, lease(2, &events));
    let work = own.append(work).expect_err("foreign append");
    let mut queue = owner(&context);
    let Err(WorkSubmitError::QueueMismatch(work)) = queue.try_submit("foreign owned work", work)
    else {
        panic!("must return foreign queue work")
    };
    assert_eq!(queue.pending(), 0);
    assert!(events.lock().unwrap().is_empty());
    drop(work);
    drop(own);
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Aborted(1),
            Event::Dropped(1),
            Event::Aborted(2),
            Event::Dropped(2)
        ]
    );
}

#[test]
#[ignore = "requires GPU"]
fn final_owner_drop_settles_work_after_observer_drop() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    {
        let mut queue = owner(&context);
        let submitted = queue
            .try_submit("owned shutdown", record(&context, lease(1, &events)))
            .unwrap();
        drop(submitted);
        assert_eq!(*events.lock().unwrap(), [Event::Submitted(1)]);
    }
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Submitted(1),
            Event::Completed(1, true),
            Event::Dropped(1)
        ]
    );
}

#[test]
#[ignore = "requires GPU"]
fn failed_callback_settles_as_failure_and_never_as_abort_or_reusable_success() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    wait_idle(&gpu.device, &gpu.queue).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let observer = Submission::pending();
    queue_failed_callback(&observer);
    let mut queue = owner(&context);
    queue.pending.push(PendingWork {
        completion: observer,
        leases: vec![Box::new(lease(1, &events))],
        failure: None,
        recovery: None,
        committed: true,
    });
    assert!(matches!(queue.collect(), Err(WaitError::Dropped)));
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Completed(1, false), Event::Dropped(1)]
    );
}

#[test]
#[ignore = "requires GPU"]
fn validated_submission_recovers_from_surface_configure_dropped_callback_once_retired() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    wait_idle(&gpu.device, &gpu.queue).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut observer = Submission::pending();
    observer.validation = Some(Arc::new(crate::wait::Validation::new(std::future::ready(
        None,
    ))));
    queue_failed_callback(&observer);
    let mut queue = owner(&context);
    queue.pending.push(PendingWork {
        completion: observer,
        leases: vec![Box::new(lease(1, &events))],
        failure: None,
        recovery: None,
        committed: false,
    });
    assert_eq!(queue.collect().unwrap(), 1);
    assert!(!queue.is_poisoned());
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Submitted(1),
            Event::Completed(1, true),
            Event::Dropped(1)
        ]
    );
}

fn queue_failed_callback(observer: &Submission) {
    drop(Notify {
        signal: Arc::clone(&observer.signal),
        timed: None,
    });
}

#[test]
#[ignore = "requires GPU"]
fn submission_panic_retains_lease_until_recovery_and_never_commits() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("issued before injected registration panic"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&buffer, 0, None);
    let work = RecordedWork::new(&context, [encoder.finish()], lease(1, &events));
    let mut queue = owner(&context);
    crate::wait::PANIC_AFTER_SUBMIT.set(true);
    assert!(matches!(
        queue.try_submit("owned actual-submit panic", work),
        Err(WorkSubmitError::SubmitPanicked(_))
    ));
    assert_eq!(queue.pending(), 1);
    assert!(
        events.lock().unwrap().is_empty(),
        "unwinding must not release or abort actually submitted work"
    );
    assert!(matches!(
        queue.shutdown(),
        Err(WaitError::SubmitPanicked(_))
    ));
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Completed(1, false), Event::Dropped(1)]
    );
}

#[test]
#[ignore = "requires GPU"]
fn captured_submission_rejection_never_commits_or_completes_as_success() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rejected with host scope"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&buffer, 0, None);
    let command = encoder.finish();
    let mut control_encoder = gpu.device.create_command_encoder(&Default::default());
    control_encoder.clear_buffer(&buffer, 0, None);
    let control = control_encoder.finish();
    buffer.destroy();
    // Negative control: raw canonical submit returns normally under a host scope,
    // even though those commands were rejected. Returning is not acceptance.
    let control_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _control = crate::submit(&gpu.queue, "raw scoped rejection control", [control]);
    assert!(crate::block_on(control_scope.pop()).is_some());
    let host_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut queue = owner(&context);
    let work = RecordedWork::new(&context, [command], lease(1, &events));
    assert!(matches!(
        queue.try_submit("owned scoped rejection", work),
        Err(WorkSubmitError::Rejected(_))
    ));
    assert!(
        events.lock().unwrap().is_empty(),
        "rejected commands must not commit progress"
    );
    assert_eq!(queue.pending(), 1);
    assert!(
        crate::block_on(host_scope.pop()).is_none(),
        "owned submit must return its captured error, not silently pass it to a host scope"
    );
    assert!(matches!(
        queue.shutdown(),
        Err(WaitError::SubmitRejected(_))
    ));
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Completed(1, false), Event::Dropped(1)]
    );
}

#[test]
fn validation_is_a_barrier_even_after_retirement_callback() {
    struct Deferred(Arc<std::sync::atomic::AtomicBool>);
    impl std::future::Future for Deferred {
        type Output = Option<wgpu::Error>;
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            if self.0.load(Ordering::SeqCst) {
                std::task::Poll::Ready(None)
            } else {
                std::task::Poll::Pending
            }
        }
    }
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut observer = Submission::pending();
    observer.validation = Some(Arc::new(crate::wait::Validation::new(Deferred(
        Arc::clone(&ready),
    ))));
    let mut notify = Notify {
        signal: Arc::clone(&observer.signal),
        timed: None,
    };
    notify.finish(Ok(()));
    assert!(
        observer.status().is_none(),
        "retirement alone does not establish valid execution"
    );
    ready.store(true, Ordering::SeqCst);
    assert!(matches!(observer.status(), Some(Ok(()))));
}

#[test]
#[ignore = "requires GPU"]
fn fatal_poll_is_latched_until_host_terminal_loss_and_does_not_repanic_on_drop() {
    let gpu = crate::shared_device().expect("shared GPU");
    let device = Arc::new(gpu.device.clone());
    let raw_queue = Arc::new(gpu.queue.clone());
    let context = WorkContext::from((Arc::clone(&device), Arc::clone(&raw_queue)));
    assert!(context.same_setup(&device, &raw_queue));
    assert!(!context.same_setup(&Arc::new(gpu.device.clone()), &raw_queue));
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut queue = owner(&context);
    let _submitted = queue
        .try_submit("owned poll fault", record(&context, lease(1, &events)))
        .unwrap();
    crate::wait::PANIC_ON_POLL.set(true);
    assert!(matches!(queue.collect(), Err(WaitError::PollPanicked(_))));
    assert_eq!(queue.pending(), 1);
    assert_eq!(*events.lock().unwrap(), [Event::Submitted(1)]);
    crate::wait::PANIC_ON_POLL.set(true);
    assert!(matches!(queue.collect(), Err(WaitError::PollPanicked(_))));
    assert!(
        crate::wait::PANIC_ON_POLL.take(),
        "latched fatal poll must not be retried"
    );
    let loss = context.loss_notifier();
    loss("simulated terminal host notification".into());
    context.mark_lost("later notification must not replace the first");
    assert_eq!(
        context.loss_reason().as_deref(),
        Some("simulated terminal host notification")
    );
    assert!(matches!(queue.shutdown(), Err(WaitError::DeviceLost(_))));
    assert_eq!(queue.pending(), 0);
    drop(queue);
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Submitted(1),
            Event::Completed(1, false),
            Event::Dropped(1)
        ]
    );
}

#[test]
#[ignore = "requires GPU"]
fn terminal_loss_notifier_has_no_strong_gpu_context_cycle() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let weak = Arc::downgrade(&context.0);
    let notify = context.loss_notifier();
    assert_eq!(Arc::strong_count(&context.0), 1);
    drop(context);
    assert!(weak.upgrade().is_none());
    notify("host callback after context teardown".into());
}

#[test]
fn checked_pending_composition_retains_values_and_harvests_every_scope() {
    use crate::wait::{Checked, ValidationState};
    use std::sync::atomic::AtomicBool;
    let ready = Arc::new(AtomicBool::new(false));
    let value_drops = Arc::new(AtomicUsize::new(0));
    let future_drops = Arc::new(AtomicUsize::new(0));
    let future_guard = Allocation(Arc::clone(&future_drops));
    let future_ready = Arc::clone(&ready);
    let delayed = Validation::new(std::future::poll_fn(move |_| {
        let _keep = &future_guard;
        if future_ready.load(Ordering::SeqCst) {
            std::task::Poll::Ready(None)
        } else {
            std::task::Poll::Pending
        }
    }));
    let rejected = Validation::Future(Mutex::new(ValidationState {
        future: None,
        outcome: Some(Err(WaitError::SubmitRejected("recording rejected".into()))),
    }));
    let mut operation = Checked {
        value: Some(Ok(Allocation(Arc::clone(&value_drops)))),
        validation: Arc::new(Validation::All(vec![Arc::new(rejected), Arc::new(delayed)])),
    };
    assert!(
        operation.poll().is_pending(),
        "an early error cannot discard other pending scopes"
    );
    assert_eq!(value_drops.load(Ordering::SeqCst), 0);
    assert_eq!(future_drops.load(Ordering::SeqCst), 0);
    ready.store(true, Ordering::SeqCst);
    assert!(matches!(
        operation.poll(),
        std::task::Poll::Ready(Err(WaitError::SubmitRejected(_)))
    ));
    assert_eq!(value_drops.load(Ordering::SeqCst), 1);
    assert_eq!(future_drops.load(Ordering::SeqCst), 1);
    assert!(matches!(
        operation.poll(),
        std::task::Poll::Ready(Err(WaitError::Dropped))
    ));
}

#[test]
#[ignore = "requires GPU"]
fn checked_native_resource_rejection_and_panic_do_not_escape_scopes() {
    let gpu = crate::shared_device().expect("shared GPU");
    let mut invalid = checked(&gpu.device, || {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid checked allocation"),
            size: 4,
            usage: wgpu::BufferUsages::empty(),
            mapped_at_creation: false,
        })
    });
    assert!(matches!(
        invalid.poll(),
        std::task::Poll::Ready(Err(WaitError::SubmitRejected(_)))
    ));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut panicked = checked(&gpu.device, || {
        let _resource = Allocation(Arc::clone(&drops));
        panic!("controlled resource operation panic");
    });
    assert!(matches!(
        panicked.poll(),
        std::task::Poll::Ready(Err(WaitError::SubmitPanicked(_)))
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let mut valid = checked(&gpu.device, || 42);
    assert!(matches!(valid.poll(), std::task::Poll::Ready(Ok(42))));
}

#[test]
#[ignore = "requires GPU"]
fn recording_rejection_aborts_before_submission() {
    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let source = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("non-copyable recording source"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let destination = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("recording destination"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let work = RecordedWork::record(
        &context,
        || {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&source, 0, &destination, 0, 4);
            encoder.finish()
        },
        lease(1, &events),
    );
    assert!(matches!(work, Err(WaitError::SubmitRejected(_))));
    assert_eq!(
        *events.lock().unwrap(),
        [Event::Aborted(1), Event::Dropped(1)]
    );
}

#[test]
fn capacity_and_wait_for_slots_zero_timeout_on_empty_and_populated_queue() {
    use std::time::Duration;

    let gpu = crate::shared_device().expect("shared GPU");
    let context = WorkContext::new(&gpu.device, &gpu.queue);
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut queue = WorkQueue::new(&context, NonZeroUsize::new(2).unwrap());

    assert_eq!(queue.capacity().get(), 2);
    assert_eq!(queue.wait_for_slots(1, Duration::ZERO).unwrap(), 0);
    assert_eq!(queue.wait_for_slots(0, Duration::ZERO).unwrap(), 0);
    assert_eq!(queue.wait_for_slots(99, Duration::ZERO).unwrap(), 0);

    let signal1: Arc<Signal> = Arc::default();
    let signal2: Arc<Signal> = Arc::default();
    queue.pending.push(PendingWork {
        completion: Submission {
            signal: Arc::clone(&signal1),
            validation: None,
        },
        leases: vec![Box::new(lease(1, &events))],
        failure: None,
        recovery: None,
        committed: true,
    });
    queue.pending.push(PendingWork {
        completion: Submission {
            signal: Arc::clone(&signal2),
            validation: None,
        },
        leases: vec![Box::new(lease(2, &events))],
        failure: None,
        recovery: None,
        committed: true,
    });
    assert_eq!(queue.pending(), 2);

    // Full queue with unresolved signals returns immediately on zero timeout.
    assert_eq!(queue.wait_for_slots(1, Duration::ZERO).unwrap(), 0);
    assert_eq!(queue.pending(), 2);

    // Retiring one entry frees one slot on the next zero-timeout wait_for_slots call.
    let mut notify1 = Notify {
        signal: signal1,
        timed: None,
    };
    notify1.finish(Ok(()));
    assert_eq!(queue.wait_for_slots(1, Duration::ZERO).unwrap(), 1);
    assert_eq!(queue.pending(), 1);

    // Clean up the remaining synthetic entry before WorkQueue::drop.
    let mut notify2 = Notify {
        signal: signal2,
        timed: None,
    };
    notify2.finish(Ok(()));
    assert_eq!(queue.wait_for_slots(2, Duration::ZERO).unwrap(), 1);
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        *events.lock().unwrap(),
        [
            Event::Completed(1, true),
            Event::Dropped(1),
            Event::Completed(2, true),
            Event::Dropped(2)
        ]
    );
}
