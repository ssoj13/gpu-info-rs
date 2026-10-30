//! Owned command buffers and their renderer reservations.
//!
//! A long-lived host context owns one bounded WorkQueue for its exact device/queue pair.
//! Normal collection polls once without waiting. Shutdown (including Drop) waits for
//! retirement on the teardown thread; there is no global orphan list or polling thread.

use super::{
    POLL_PERIOD, Submission, Validation, WaitError, checked, panic_message, poll, submit_checked,
};
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, PoisonError};

/// An adopted host device/queue pair and its non-forgeable ownership identity.
///
/// Create this once from the matching pair returned by the host's device creation,
/// then share clones with producers and WorkQueues. Independently adopted contexts
/// are conservatively distinct, even when their raw handles happen to compare equal.
/// wgpu native handle IDs can collide across Instances, so raw handle equality is
/// deliberately not used as an ownership proof.
#[derive(Clone)]
pub struct WorkContext(Arc<Context>);

struct Context {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    loss: Arc<std::sync::Mutex<Option<String>>>,
    memory: crate::budget::MemoryBudget,
    resources: crate::resources::Resources,
}

impl WorkContext {
    /// Adopt the host's matching pair without creating or replacing any GPU device.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from((Arc::new(device.clone()), Arc::new(queue.clone())))
    }

    /// Report terminal device loss from the host's existing loss callback.
    /// This is a loss proof, not a general cancellation request.
    pub fn mark_lost(&self, reason: impl Into<String>) {
        let mut loss = self.0.loss.lock().unwrap_or_else(PoisonError::into_inner);
        if loss.is_none() {
            *loss = Some(reason.into());
        }
    }

    /// A callback adapter retaining only weak CPU notification state, never the GPU device.
    /// Chain it into the existing host device-loss callback instead of replacing that callback.
    pub fn loss_notifier(&self) -> impl Fn(String) + Send + Sync + 'static {
        let loss = Arc::downgrade(&self.0.loss);
        move |reason| {
            if let Some(loss) = loss.upgrade() {
                let mut loss = loss.lock().unwrap_or_else(PoisonError::into_inner);
                if loss.is_none() {
                    *loss = Some(reason);
                }
            }
        }
    }

    /// The first terminal loss reported by the host.
    pub fn loss_reason(&self) -> Option<String> {
        self.0
            .loss
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The exact Arc-backed origin accepted by the host/Burn registration.
    pub fn device_arc(&self) -> &Arc<wgpu::Device> {
        &self.0.device
    }

    /// The exact Arc-backed queue origin.
    pub fn queue_arc(&self) -> &Arc<wgpu::Queue> {
        &self.0.queue
    }

    /// Verify host-origin Arcs without relying on backend-local raw handle IDs.
    pub fn same_setup(&self, device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> bool {
        Arc::ptr_eq(&self.0.device, device) && Arc::ptr_eq(&self.0.queue, queue)
    }

    /// Shared logical allocation admission for this exact adopted context.
    /// Configure its allowance before allocating tracked resources. This does
    /// not include foreign allocations or replace driver headroom observation.
    pub fn memory(&self) -> &crate::budget::MemoryBudget {
        &self.0.memory
    }

    /// Shared tracked allocations and reuse for this exact ownership domain.
    pub fn resources(&self) -> &crate::resources::Resources {
        &self.0.resources
    }

    /// The adopted device, for internal producer recording.
    pub fn device(&self) -> &wgpu::Device {
        &self.0.device
    }

    /// The adopted queue, for uploads and the canonical submission owner.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.0.queue
    }
}

impl From<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> for WorkContext {
    fn from((device, queue): (Arc<wgpu::Device>, Arc<wgpu::Queue>)) -> Self {
        let memory = crate::budget::MemoryBudget::default();
        let loss = Arc::default();
        let resources =
            crate::resources::Resources::new(device.clone(), memory.clone(), Arc::clone(&loss));
        Self(Arc::new(Context {
            device,
            queue,
            loss,
            memory,
            resources,
        }))
    }
}

impl PartialEq for WorkContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for WorkContext {}

impl std::fmt::Debug for WorkContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkContext").finish_non_exhaustive()
    }
}

/// A renderer reservation and the resources/byte charges its commands may observe.
///
/// Hooks must not panic or re-enter their WorkQueue. `submitted` runs synchronously
/// after queue submission and its error scopes succeed. Native scope futures are
/// immediate; asynchronous backends defer this hook to owner collection.
/// `completed` runs only during owner collection,
/// outside wgpu callbacks and signal locks. A failed outcome must never recycle a
/// resource as a successful result. Do not retain the WorkQueue strongly from a lease.
pub trait WorkLease: Send + 'static {
    /// Commit recording-local state after actual submission.
    fn submitted(&mut self) {}

    /// Settle retained resources after known retirement.
    fn completed(self: Box<Self>, _outcome: Result<(), WaitError>) {}

    /// Release an unsubmitted recording without committing its state.
    fn aborted(self: Box<Self>) {}
}

impl WorkLease for () {}

/// Commands and their reservations, moved together into a WorkQueue.
///
/// Producers finish their private encoders internally and immediately transfer the
/// resulting command buffers and lease here. There is deliberately no raw extractor
/// or mutable encoder accessor.
#[must_use = "submit the recorded work, or drop it to abort the recording"]
pub struct RecordedWork {
    context: WorkContext,
    commands: Vec<wgpu::CommandBuffer>,
    leases: Vec<Box<dyn WorkLease>>,
    validation: Vec<Arc<Validation>>,
}

impl std::fmt::Debug for RecordedWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordedWork")
            .field("commands", &self.commands.len())
            .field("leases", &self.leases.len())
            .finish_non_exhaustive()
    }
}

impl RecordedWork {
    /// Transfer already recorded commands from a producer on the adopted host context.
    ///
    /// This unscoped interop entry point cannot validate earlier resource creation
    /// or encoding. Its caller owns those checks. Prefer Self::record.
    pub fn new(
        context: &WorkContext,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
        lease: impl WorkLease,
    ) -> Self {
        Self {
            context: context.clone(),
            commands: commands.into_iter().collect(),
            leases: vec![Box::new(lease)],
            validation: Vec::new(),
        }
    }

    /// Encode a command buffer under owned validation/OOM/internal scopes.
    /// Native rejection returns before queue submission. Asynchronous scopes travel
    /// with the work and gate every accepted hook in the receiving WorkQueue.
    pub fn record(
        context: &WorkContext,
        operation: impl FnOnce() -> wgpu::CommandBuffer,
        lease: impl WorkLease,
    ) -> Result<Self, WaitError> {
        let mut operation = checked(context.device(), operation);
        let mut work = Self::new(context, [], lease);
        work.validation.push(Arc::clone(&operation.validation));
        work.commands
            .push(operation.value.take().expect("owned operation result")?);
        if let Some(Err(error)) = Validation::poll_all(
            &work.validation,
            &mut std::task::Context::from_waker(std::task::Waker::noop()),
        ) {
            return Err(error);
        }
        Ok(work)
    }

    /// Append work in command order. A foreign queue returns the other work unchanged.
    pub fn append(&mut self, mut other: Self) -> Result<(), Self> {
        if self.context != other.context {
            return Err(other);
        }
        self.commands.append(&mut other.commands);
        self.leases.append(&mut other.leases);
        self.validation.append(&mut other.validation);
        Ok(())
    }
}

impl Drop for RecordedWork {
    fn drop(&mut self) {
        // Discard unsent commands before releasing the reservations they referenced.
        self.commands.clear();
        for lease in self.leases.drain(..) {
            lease.aborted();
        }
    }
}

/// A submission rejected before consumption returns the complete original work.
#[derive(Debug, thiserror::Error)]
pub enum WorkSubmitError {
    /// The bounded owner has no free submission slot.
    #[error("GPU work queue is full")]
    Busy(RecordedWork),
    /// Commands were recorded for a different exact queue.
    #[error("recorded work belongs to another GPU queue")]
    QueueMismatch(RecordedWork),
    /// An earlier indeterminate submission has not been proved retired.
    #[error("GPU work queue is recovering an indeterminate submission")]
    Poisoned(RecordedWork),
    /// Polling failed before this work was submitted.
    #[error("GPU work collection failed: {error}")]
    Poll {
        /// The collection error.
        error: WaitError,
        /// The unchanged unsubmitted work.
        work: RecordedWork,
    },
    /// Commands might have been submitted: the owner retains their leases for recovery.
    #[error("GPU submission panicked: {0}")]
    SubmitPanicked(String),
    /// Submission was rejected by its owned validation/memory/internal error scopes.
    #[error("GPU submission rejected: {0}")]
    Rejected(String),
}

struct PendingWork {
    completion: Submission,
    leases: Vec<Box<dyn WorkLease>>,
    failure: Option<WaitError>,
    recovery: Option<Submission>,
    committed: bool,
}

impl PendingWork {
    /// The submission whose completion signal is currently awaited for retirement (`recovery`
    /// once a failed submission has queued a recovery fence, otherwise `completion`).
    fn active_submission(&self) -> &Submission {
        self.recovery.as_ref().unwrap_or(&self.completion)
    }

    /// Poll both original and recovery validation scopes without short-circuiting and report
    /// whether all owned validation scopes have settled.
    fn validations_settled(&self) -> bool {
        let completion_settled = self
            .completion
            .validation
            .as_ref()
            .is_none_or(|validation| validation.status().is_some());
        let recovery_settled = self.recovery.as_ref().is_none_or(|recovery| {
            recovery
                .validation
                .as_ref()
                .is_none_or(|validation| validation.status().is_some())
        });
        completion_settled && recovery_settled
    }

    fn submitted(&mut self) {
        self.committed = true;
        for lease in &mut self.leases {
            lease.submitted();
        }
    }

    fn finish(self, outcome: Result<(), WaitError>) {
        for lease in self.leases {
            lease.completed(outcome.clone());
        }
    }
}

/// Bounded retained submissions for one adopted host device and its exact queue.
///
/// The host must pass the matching device/queue pair and keep this owner alive for
/// normal rendering. It never chooses a device or uses the canonical compute pool.
/// Capacity bounds submitted metadata; renderer leases own unique byte accounting.
/// Before submitting, chain WorkContext::loss_notifier into the host's existing
/// device-loss callback. It retains only weak CPU state and never owns the device.
///
/// `collect` never waits. `shutdown` and Drop wait through the canonical nonblocking
/// polling loop, so final owner teardown belongs on a worker/teardown thread. As with
/// `wait`, an unresponsive GPU can delay teardown; unknown retirement is never
/// interpreted as permission to release/reuse attachments.
/// Browser event loops cannot be blocked: WebGPU consumers must collect all work
/// asynchronously before owner teardown, just as they must not call blocking `wait`.
pub struct WorkQueue {
    context: WorkContext,
    pending: Vec<PendingWork>,
    capacity: NonZeroUsize,
    poisoned: bool,
    poll_fault: Option<WaitError>,
}

impl std::fmt::Debug for WorkQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkQueue")
            .field("pending", &self.pending.len())
            .field("capacity", &self.capacity)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl WorkQueue {
    /// Create one owner for a matching adopted host device/queue pair.
    pub fn new(context: &WorkContext, capacity: NonZeroUsize) -> Self {
        Self {
            context: context.clone(),
            pending: Vec::with_capacity(capacity.get()),
            capacity,
            poisoned: false,
            poll_fault: None,
        }
    }

    /// The shared adopted-context capability for producers and other bounded owners.
    pub fn context(&self) -> &WorkContext {
        &self.context
    }

    /// Number of submitted reservations still retained, including completed ones not collected.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Maximum number of retained submissions before [`Self::try_submit`] reports `Busy`.
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    /// Whether a failed submission still needs a retirement proof.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Submit owned work without waiting for capacity or GPU completion.
    ///
    /// The pending owner is installed before the driver call. Unwinding during
    /// submission therefore cannot turn possibly submitted leases into aborts.
    pub fn try_submit(
        &mut self,
        label: &'static str,
        mut work: RecordedWork,
    ) -> Result<Submission, WorkSubmitError> {
        if work.context != self.context {
            return Err(WorkSubmitError::QueueMismatch(work));
        }
        if let Err(error) = self.collect() {
            return Err(WorkSubmitError::Poll { error, work });
        }
        if self.poisoned {
            return Err(WorkSubmitError::Poisoned(work));
        }
        if self.pending.len() == self.capacity.get() {
            return Err(WorkSubmitError::Busy(work));
        }

        let observer = Submission::pending();
        let commands = std::mem::take(&mut work.commands);
        let mut validations = std::mem::take(&mut work.validation);
        // A known encoding rejection is a true abort: nothing reached the queue.
        if let Some(Err(error)) = Validation::poll_all(
            &validations,
            &mut std::task::Context::from_waker(std::task::Waker::noop()),
        ) {
            return Err(WorkSubmitError::Rejected(error.to_string()));
        }
        let index = self.pending.len();
        self.pending.push(PendingWork {
            completion: observer.clone(),
            leases: std::mem::take(&mut work.leases),
            failure: None,
            recovery: None,
            committed: false,
        });
        let signal = Arc::clone(&observer.signal);
        let result = catch_unwind(AssertUnwindSafe(|| {
            submit_checked(
                self.context.device(),
                self.context.queue(),
                label,
                commands,
                signal,
            )
        }));
        let result = match result {
            Ok(mut operation) => {
                validations.push(operation.validation);
                operation.value.take().expect("owned submission result")
            }
            Err(payload) => Err(WaitError::SubmitPanicked(panic_message(payload.as_ref()))),
        };
        let mut submission = result.as_ref().cloned().unwrap_or(observer);
        submission.validation = Some(Arc::new(Validation::All(validations)));
        self.pending[index].completion = submission.clone();
        match result {
            Ok(_) => {
                match submission.validation_status() {
                    Some(Ok(())) => self.pending[index].submitted(),
                    Some(Err(error)) => {
                        let message = error.to_string();
                        self.pending[index].failure = Some(error);
                        self.poisoned = true;
                        self.recover(index);
                        return Err(WorkSubmitError::Rejected(message));
                    }
                    None => {}
                }
                Ok(submission)
            }
            Err(error) => {
                let message = match &error {
                    WaitError::SubmitPanicked(message) => message.clone(),
                    _ => error.to_string(),
                };
                self.pending[index].failure = Some(error);
                self.poisoned = true;
                self.recover(index);
                Err(WorkSubmitError::SubmitPanicked(message))
            }
        }
    }

    /// Poll once, then settle terminal reservations outside callbacks and signal locks.
    ///
    /// An error is reported only after known-retired failed leases have been settled,
    /// except a poll error, which leaves every unproved lease owned for retry/shutdown.
    pub fn collect(&mut self) -> Result<usize, WaitError> {
        if let Some(error) = self.terminal_loss() {
            return Err(error);
        }
        if let Some(error) = &self.poll_fault {
            return Err(error.clone());
        }
        let polled = poll(self.context.device());
        if let Some(error) = self.terminal_loss() {
            return Err(error);
        }
        let status = match polled {
            Err(error @ WaitError::PollPanicked(_)) => {
                self.poll_fault = Some(error.clone());
                self.poisoned = true;
                return Err(error);
            }
            other => other?,
        };
        // WebGPU poll is a no-op returning QueueEmpty even while its promise is pending.
        let queue_empty = status.is_queue_empty() && !cfg!(target_arch = "wasm32");
        let mut completed = 0;
        let mut failure = None;
        let mut index = 0;
        while index < self.pending.len() {
            let validation = self.pending[index].completion.validation_status();
            if !self.pending[index].committed
                && self.pending[index].failure.is_none()
                && matches!(validation, Some(Ok(())))
            {
                self.pending[index].submitted();
            }
            let status = self.pending[index].completion.status();
            if self.pending[index].failure.is_none()
                && let Some(Err(error)) = &status
            {
                self.pending[index].failure = Some(error.clone());
                self.poisoned = true;
                if !queue_empty {
                    self.recover(index);
                }
            }

            let entry = &self.pending[index];
            let outcome = if let Some(error) = &entry.failure {
                if entry.validations_settled()
                    && (queue_empty
                        || entry
                            .recovery
                            .as_ref()
                            .is_some_and(|recovery| matches!(recovery.status(), Some(Ok(())))))
                {
                    Some(Err(error.clone()))
                } else {
                    None
                }
            } else {
                status
            };
            if let Some(outcome) = outcome {
                if let Err(error) = &outcome {
                    failure.get_or_insert_with(|| error.clone());
                }
                // Preserve strict FIFO submission order so `self.pending.first()` is always
                // the oldest in-flight GPU submission for `wait_for_slots` and `pause`.
                let entry = self.pending.remove(index);
                entry.finish(outcome);
                completed += 1;
            } else {
                index += 1;
            }
        }
        self.poisoned = self.pending.iter().any(|entry| entry.failure.is_some());
        failure.map_or(Ok(completed), Err)
    }

    /// Poll and wait up to `timeout` until at least `needed` submission slots are available.
    ///
    /// `needed` is clamped to `1..=self.capacity().get()`. Returns the total number of
    /// submissions retired during this call, or returns early once `timeout` elapses or
    /// the queue becomes poisoned.
    ///
    /// Each wait slice is bounded by [`POLL_PERIOD`] (1 ms) with the signal mutex dropped
    /// before `self.collect()` calls `device.poll(wgpu::PollType::Poll)`, ensuring that
    /// `wgpu::Queue::on_submitted_work_done` callbacks are driven even when the host GUI
    /// thread is parked by event-driven `GuiWaker`.
    pub fn wait_for_slots(
        &mut self,
        needed: usize,
        timeout: std::time::Duration,
    ) -> Result<usize, WaitError> {
        let needed = needed.clamp(1, self.capacity.get());
        let max_pending = self.capacity.get() - needed;
        let mut total_completed = self.collect()?;
        if self.pending.len() <= max_pending || self.poisoned || timeout.is_zero() {
            return Ok(total_completed);
        }
        let start = std::time::Instant::now();
        let deadline = start.checked_add(timeout);
        loop {
            if self.pending.len() <= max_pending || self.poisoned {
                break;
            }
            let now = std::time::Instant::now();
            let remaining = match deadline {
                Some(deadline) => {
                    if now >= deadline {
                        break;
                    }
                    deadline - now
                }
                None => timeout.saturating_sub(now.saturating_duration_since(start)),
            };
            if remaining.is_zero() {
                break;
            }
            let slice = remaining.min(POLL_PERIOD);
            self.wait_slice(slice);
            total_completed += self.collect()?;
        }
        Ok(total_completed)
    }

    /// Wait for and settle all retained work on a worker/teardown thread.
    ///
    /// A poll error preserves pending leases and returns it to the caller. Keep the
    /// owner alive and route terminal loss through its host notification. Fatal polling
    /// is latched, never repeated during unwinding or interpreted as retirement.
    pub fn shutdown(&mut self) -> Result<(), WaitError> {
        let mut failure = None;
        while !self.pending.is_empty() {
            match self.collect() {
                Ok(_) => {}
                Err(error) => {
                    if !self.pending.is_empty()
                        && matches!(error, WaitError::Poll(_) | WaitError::PollPanicked(_))
                    {
                        return Err(error);
                    }
                    failure.get_or_insert(error);
                }
            }
            if !self.pending.is_empty() {
                self.pause();
            }
        }
        failure.map_or(Ok(()), Err)
    }

    fn recover(&mut self, index: usize) {
        // Keep the original scope owner intact. A later fence proves retirement,
        // while both original and recovery validations must still be harvested.
        if let Ok(mut operation) = catch_unwind(AssertUnwindSafe(|| {
            submit_checked(
                self.context.device(),
                self.context.queue(),
                "gpu-info submission recovery",
                [],
                Arc::default(),
            )
        })) {
            if let Some(Ok(mut fence)) = operation.value.take() {
                fence.validation = Some(operation.validation);
                self.pending[index].recovery = Some(fence);
            } else {
                // Even a rejected recovery operation owns scope futures.
                let mut observer = Submission::pending();
                observer.validation = Some(operation.validation);
                self.pending[index].recovery = Some(observer);
            }
        }
    }

    fn terminal_loss(&mut self) -> Option<WaitError> {
        let error = WaitError::DeviceLost(self.context.loss_reason()?);
        self.poisoned = true;
        let pending = std::mem::take(&mut self.pending);
        for entry in pending {
            entry.finish(Err(error.clone()));
        }
        Some(error)
    }

    /// Wait up to `slice` on the oldest in-flight submission's condition variable.
    ///
    /// Invariant: validation futures are polled before locking the signal mutex (avoiding
    /// nested lock order between `Validation` and `Signal`), and the signal mutex guard is
    /// dropped before returning so the subsequent `self.collect()` -> `device.poll(PollType::Poll)`
    /// call can never deadlock when `wgpu` fires `on_submitted_work_done` synchronously.
    fn wait_slice(&self, slice: std::time::Duration) {
        if let Some(entry) = self.pending.first() {
            let validations_settled = entry.validations_settled();
            let active = entry.active_submission();
            let (state, changed) = &*active.signal;
            let guard = state.lock().unwrap_or_else(PoisonError::into_inner);
            if guard.is_none() || !validations_settled || entry.failure.is_some() {
                let _slept = changed
                    .wait_timeout(guard, slice)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
    }

    fn pause(&self) {
        self.wait_slice(POLL_PERIOD);
    }
}

impl Drop for WorkQueue {
    fn drop(&mut self) {
        let mut reported = false;
        while !self.pending.is_empty() {
            if let Err(error) = self.shutdown()
                && !self.pending.is_empty()
            {
                if !reported {
                    log::error!(
                        "gpu-info: work-queue shutdown failed; retaining pending work: {error}"
                    );
                    reported = true;
                }
                self.pause();
            }
        }
    }
}

#[cfg(test)]
#[path = "work_tests.rs"]
mod tests;

#[cfg(test)]
mod slot_wait_tests {
    use super::*;
    use crate::wait::{Notify, Signal};
    use std::time::{Duration, Instant};

    #[test]
    fn active_submission_and_validations_settled_track_recovery_and_scopes() {
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&ready);
        let mut completion = Submission::pending();
        completion.validation = Some(Arc::new(Validation::new(std::future::poll_fn(move |_| {
            if flag.load(std::sync::atomic::Ordering::SeqCst) {
                std::task::Poll::Ready(None)
            } else {
                std::task::Poll::Pending
            }
        }))));

        let recovery = Submission::pending();
        let recovery_signal = Arc::clone(&recovery.signal);
        let mut entry = PendingWork {
            completion,
            leases: Vec::new(),
            failure: None,
            recovery: None,
            committed: true,
        };

        assert!(!entry.validations_settled());
        assert!(!Arc::ptr_eq(&entry.active_submission().signal, &recovery_signal));

        entry.recovery = Some(recovery);
        assert!(Arc::ptr_eq(&entry.active_submission().signal, &recovery_signal));
        assert!(!entry.validations_settled());

        ready.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(entry.validations_settled());
    }

    #[test]
    fn collect_preserves_fifo_order_and_wait_for_slots_exercises_nonzero_timeout() {
        let Some(gpu) = crate::shared_device() else {
            return;
        };
        let context = WorkContext::new(&gpu.device, &gpu.queue);
        let mut queue = WorkQueue::new(&context, NonZeroUsize::new(3).unwrap());

        let signal1: Arc<Signal> = Arc::default();
        let signal2: Arc<Signal> = Arc::default();
        let signal3: Arc<Signal> = Arc::default();
        for signal in [&signal1, &signal2, &signal3] {
            queue.pending.push(PendingWork {
                completion: Submission {
                    signal: Arc::clone(signal),
                    validation: None,
                },
                leases: vec![Box::new(())],
                failure: None,
                recovery: None,
                committed: true,
            });
        }

        // Retiring the oldest entry (index 0) must keep [signal2, signal3] in FIFO order,
        // never [signal3, signal2] as `swap_remove(0)` did.
        Notify {
            signal: Arc::clone(&signal1),
            timed: None,
        }
        .finish(Ok(()));
        assert_eq!(queue.wait_for_slots(1, Duration::ZERO).unwrap(), 1);
        assert_eq!(queue.pending(), 2);
        assert!(
            Arc::ptr_eq(&queue.pending[0].completion.signal, &signal2),
            "pending[0] must remain the oldest unresolved submission (signal2)"
        );

        // Non-zero timeout expires cleanly when 2 free slots are requested (`max_pending = 1`)
        // and `signal2` has not completed yet.
        assert_eq!(
            queue
                .wait_for_slots(2, Duration::from_millis(3))
                .unwrap(),
            0
        );
        assert_eq!(queue.pending(), 2);

        // When `signal2` (the oldest remaining submission) is notified on a worker thread,
        // `wait_for_slots(2, ...)` wakes on `pending[0]` (`signal2`) well before the 500 ms timeout.
        let signal2_worker = Arc::clone(&signal2);
        let notifier = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            Notify {
                signal: signal2_worker,
                timed: None,
            }
            .finish(Ok(()));
        });
        let started = Instant::now();
        assert_eq!(
            queue
                .wait_for_slots(2, Duration::from_millis(500))
                .unwrap(),
            1
        );
        assert!(started.elapsed() < Duration::from_millis(250));
        assert_eq!(queue.pending(), 1);
        notifier.join().unwrap();

        // Retire `signal3` before drop.
        Notify {
            signal: signal3,
            timed: None,
        }
        .finish(Ok(()));
        assert_eq!(queue.wait_for_slots(3, Duration::ZERO).unwrap(), 1);
        assert_eq!(queue.pending(), 0);
    }
}
