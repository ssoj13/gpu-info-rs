//! The ONE way the cluster submits GPU work it waits for and waits for it: [`submit`] then [`wait`](fn@crate::wait::wait)
//! (or [`wait_idle`]), and [`map_read`] for a readback. No wait ever blocks inside wgpu.
//!
//! **Why never a blocking `Device::poll`:** wgpu-core 30 `Device::maintain` holds the device's
//! snatch lock for reading across its fence wait (`device/resource.rs:797-879`), and
//! `Surface::present` (`present.rs:340`), `Buffer::destroy` and `unmap` take it for writing: one
//! thread waiting for its readback froze the UI's `present` for as long as the GPU worked (Playa:
//! 1.8 s, ofx-rs plan0.md 3.H5).
//!
//! **Why not a blocking poll with a timeout either:** `maintain` reads the fence value, then retires
//! finished submissions; `Queue::submit` runs `maintain(Poll)` inline (`device/queue.rs:1541`), so
//! another thread can retire the awaited submission in between. The timed-out waiter then finds the
//! queue empty below the index it waited for and hits `assert!` at `device/resource.rs:948` (seen
//! twice in Playa). Only `PollType::Poll` has no index to assert on: the assertion sits inside
//! `if let Some(wait_submission_index)` (`device/resource.rs:945`), which `Poll` never enters. That
//! is the guarantee, by construction; a stress test (2 waiters of ~1 ms work beside 2 foreign
//! pollers, 6000 waits) did not reproduce the race with the old timed wait either, so none is kept.
//!
//! **So:** completion is a callback. [`submit`] registers `Queue::on_submitted_work_done` right
//! after its submission, both under one process-wide lock, so the callback belongs to that
//! submission (it attaches to the last tracked submission, `device/life.rs:372-388`; only a
//! submission made outside this module can slip in between, which makes the wait longer, never
//! shorter). The waiter polls without blocking (callbacks run inside `poll` and inside every
//! thread's `submit`) and sleeps on a condition variable for [`POLL_PERIOD`] between polls, woken
//! early when any thread's poll ran its callback. `Condvar::wait_timeout(1 ms)` measured 1.46 ms
//! median, 2.5 ms max on Windows 11 (2026-09-28).
//!
//! **Why a mutex and condition variable, not a channel or a park executor:** plug-ins call this on
//! host threads. A blocking `std::sync::mpsc` receive and `std::thread::park` both reach
//! `std::thread::current()`, which on glibc registers a thread-exit destructor inside the calling
//! module that `dlclose` does not unregister (ofx-rs PLAN 2.5). [`block_on`] uses `pollster`,
//! which waits the same way.
//!
//! **Every submission is labelled and counted** ([`submit_stats`]): per label, how many, how many
//! are in flight, and the time from `submit` to the observed completion (queueing behind other work
//! included, so a label whose time is long while its own work is short names what it waits behind).
//! A submission slower than [`SLOW_SUBMISSION`] is logged at debug level. What finds the work that
//! holds a device's queue - and so an application's frames - without timestamp queries.
//!
//! **Order for callers with error scopes:** call [`map_read`], then pop the scopes, then look at
//! the result: a scope that captured an out-of-memory error still names the real cause of a
//! failed map.

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The longest a waiter sleeps between two non-blocking polls when no other thread's poll wakes it.
pub const POLL_PERIOD: Duration = Duration::from_millis(1);

/// Why a wait failed. Every variant means the awaited work's results must not be read.
#[derive(Debug, thiserror::Error)]
pub enum WaitError {
    /// [`wgpu::Device::poll`] reported a polling error.
    #[error("device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    /// wgpu completed a buffer map with an error.
    #[error("buffer map failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    /// wgpu dropped the completion callback without calling it (the device or buffer is gone).
    #[error("completion callback dropped without running")]
    Dropped,
    /// Submission unwound; a recovery fence, not the failed work, establishes retirement.
    #[error("GPU submission panicked: {0}")]
    SubmitPanicked(String),
    /// Submission was rejected by one of its owned error scopes.
    #[error("GPU submission rejected: {0}")]
    SubmitRejected(String),
    /// Native polling unwound; this is not a retirement proof.
    #[error("GPU polling panicked: {0}")]
    PollPanicked(String),
    /// The host reported terminal loss of the adopted context.
    #[error("GPU device lost: {0}")]
    DeviceLost(String),
}

impl Clone for WaitError {
    fn clone(&self) -> Self {
        match self {
            Self::Poll(wgpu::PollError::Timeout) => Self::Poll(wgpu::PollError::Timeout),
            Self::Poll(wgpu::PollError::WrongSubmissionIndex(a, b)) => {
                Self::Poll(wgpu::PollError::WrongSubmissionIndex(*a, *b))
            }
            Self::Map(error) => Self::Map(error.clone()),
            Self::Dropped => Self::Dropped,
            Self::SubmitPanicked(message) => Self::SubmitPanicked(message.clone()),
            Self::SubmitRejected(message) => Self::SubmitRejected(message.clone()),
            Self::PollPanicked(message) => Self::PollPanicked(message.clone()),
            Self::DeviceLost(message) => Self::DeviceLost(message.clone()),
        }
    }
}

mod work;
pub use work::{RecordedWork, WorkContext, WorkLease, WorkQueue, WorkSubmitError};

/// A submission slower than this, from `submit` to its observed completion, is logged at debug level.
pub const SLOW_SUBMISSION: Duration = Duration::from_millis(50);

/// The submissions of one label ([`submit_stats`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmitStats {
    /// The label [`submit`] was given.
    pub label: &'static str,
    /// Submissions made.
    pub count: u64,
    /// Submitted and not yet observed complete.
    pub in_flight: u64,
    /// Summed time from `submit` to observed completion, of the completed ones.
    pub total: Duration,
    /// The longest of them.
    pub max: Duration,
}

/// Every label's [`SubmitStats`], in first-submission order.
static STATS: Mutex<Vec<SubmitStats>> = Mutex::new(Vec::new());

/// A snapshot of every label's [`SubmitStats`] since the process started.
pub fn submit_stats() -> Vec<SubmitStats> {
    STATS.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Run `update` on `label`'s stats, creating them at the first submission.
fn with_stats(label: &'static str, update: impl FnOnce(&mut SubmitStats)) {
    let mut stats = STATS.lock().unwrap_or_else(PoisonError::into_inner);
    let index = match stats.iter().position(|entry| entry.label == label) {
        Some(index) => index,
        None => {
            stats.push(SubmitStats {
                label,
                count: 0,
                in_flight: 0,
                total: Duration::ZERO,
                max: Duration::ZERO,
            });
            stats.len() - 1
        }
    };
    if let Some(entry) = stats.get_mut(index) {
        update(entry);
    }
}

/// Work submitted through [`submit`]; [`wait`] blocks until it completed.
#[derive(Clone)]
#[must_use = "wait for it, or drop it deliberately: nothing else tells when it completed"]
pub struct Submission {
    /// Set by the submission's completion callback.
    signal: Arc<Signal>,
    validation: Option<Arc<Validation>>,
}

trait ValidationFuture: std::future::Future<Output = Option<wgpu::Error>> + wgpu::WasmNotSend {}
impl<T> ValidationFuture for T where
    T: std::future::Future<Output = Option<wgpu::Error>> + wgpu::WasmNotSend
{
}

struct ValidationState {
    future: Option<std::pin::Pin<Box<dyn ValidationFuture>>>,
    outcome: Option<Result<(), WaitError>>,
}

/// Error scopes are popped on the producing thread; only their futures travel.
enum Validation {
    Future(Mutex<ValidationState>),
    All(Vec<Arc<Validation>>),
}

impl Validation {
    fn new(future: impl ValidationFuture + 'static) -> Self {
        Self::Future(Mutex::new(ValidationState {
            future: Some(Box::pin(future)),
            outcome: None,
        }))
    }

    fn poll_all(
        validations: &[Arc<Self>],
        cx: &mut std::task::Context<'_>,
    ) -> Option<Result<(), WaitError>> {
        let mut pending = false;
        let mut failure = None;
        for validation in validations {
            match validation.poll(cx) {
                None => pending = true,
                Some(Err(error)) if failure.is_none() => failure = Some(error),
                Some(_) => {}
            }
        }
        (!pending).then(|| failure.map_or(Ok(()), Err))
    }

    fn status(&self) -> Option<Result<(), WaitError>> {
        self.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    }

    fn poll(&self, cx: &mut std::task::Context<'_>) -> Option<Result<(), WaitError>> {
        let mutex = match self {
            Self::All(validations) => return Self::poll_all(validations, cx),
            Self::Future(mutex) => mutex,
        };
        let mut future = {
            let mut state = mutex.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(outcome) = &state.outcome {
                return Some(outcome.clone());
            }
            state.future.take()?
        };
        // Poll and destroy backend futures outside the metadata mutex.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| future.as_mut().poll(cx)));
        let outcome = match result {
            Ok(std::task::Poll::Ready(error)) => Some(error.map_or(Ok(()), |error| {
                Err(WaitError::SubmitRejected(error.to_string()))
            })),
            Ok(std::task::Poll::Pending) => None,
            Err(payload) => Some(Err(WaitError::SubmitRejected(format!(
                "error-scope polling panicked: {}",
                panic_message(payload.as_ref())
            )))),
        };
        let mut state = mutex.lock().unwrap_or_else(PoisonError::into_inner);
        state.outcome = outcome.clone();
        if outcome.is_none() {
            state.future = Some(future);
        }
        outcome
    }
}

/// A resource operation whose result remains private until all owned error scopes settle.
/// No GPU submission or completion wait is performed by this owner.
#[must_use = "poll or resolve the checked operation before using its result"]
pub struct Checked<T> {
    value: Option<Result<T, WaitError>>,
    validation: Arc<Validation>,
}

impl<T> std::fmt::Debug for Checked<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Checked")
            .field("has_result", &self.value.is_some())
            .finish_non_exhaustive()
    }
}

impl<T> Checked<T> {
    /// Inspect validation without blocking. Pending retains both result and futures.
    /// Ready consumes the result; subsequent polls return Dropped.
    pub fn poll(&mut self) -> std::task::Poll<Result<T, WaitError>> {
        self.poll_with(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    }

    fn poll_with(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<T, WaitError>> {
        if self.value.is_none() {
            return std::task::Poll::Ready(Err(WaitError::Dropped));
        }
        let Some(validation) = self.validation.poll(cx) else {
            return std::task::Poll::Pending;
        };
        let value = self.value.take().unwrap_or(Err(WaitError::Dropped));
        std::task::Poll::Ready(validation.and(value))
    }

    /// Await validation while retaining the operation result and every scope future.
    pub async fn resolve(mut self) -> Result<T, WaitError> {
        std::future::poll_fn(|cx| self.poll_with(cx)).await
    }
}

/// Run one resource/recording operation under validation, OOM and internal scopes.
/// Scopes pop on this thread even if the operation panics; Pending is an owned result,
/// never proof that the operation succeeded. Use RecordedWork::record for GPU commands.
pub fn checked<T>(device: &wgpu::Device, operation: impl FnOnce() -> T) -> Checked<T> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation))
        .map_err(|payload| WaitError::SubmitPanicked(panic_message(payload.as_ref())));
    let internal = internal.pop();
    let memory = memory.pop();
    let validation = validation.pop();
    Checked {
        value: Some(value),
        validation: Arc::new(Validation::new(async move {
            let internal = internal.await;
            let memory = memory.await;
            let validation = validation.await;
            internal.or(memory).or(validation)
        })),
    }
}

fn outcome(signal: &Signal, validation: Option<&Validation>) -> Option<Result<(), WaitError>> {
    if let Some(validation) = validation {
        match validation.status()? {
            Ok(()) => {}
            Err(error) => return Some(Err(error)),
        }
    }
    signal
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

impl Submission {
    /// The observed outcome without GPU polling, waiting or consuming it.
    /// Owned WorkQueue submissions also poll their CPU error-scope futures here.
    /// Plain submit reports retirement only: its caller must still check host error scopes.
    /// A failure does not itself prove retirement; WorkQueue retains failed leases until proof.
    pub fn status(&self) -> Option<Result<(), WaitError>> {
        outcome(&self.signal, self.validation.as_deref())
    }

    fn pending() -> Self {
        Self {
            signal: Arc::default(),
            validation: None,
        }
    }

    /// Whether owned submission error scopes have accepted its commands.
    ///
    /// `Some(Ok(()))` permits dependent work; it does not prove GPU retirement.
    /// `None` means validation is pending, or this came from plain [`submit`]
    /// without owned error scopes. Inspect [`Self::status`] for completion.
    pub fn validation_status(&self) -> Option<Result<(), WaitError>> {
        self.validation
            .as_ref()
            .and_then(|validation| validation.status())
    }
}

/// Submit `commands` on `queue` as `label` (what the work is, for [`submit_stats`]) and register the
/// submission's completion callback (see the module docs). The only submit of the cluster.
pub fn submit<I>(queue: &wgpu::Queue, label: &'static str, commands: I) -> Submission
where
    I: IntoIterator<Item = wgpu::CommandBuffer>,
{
    submit_to(queue, label, commands, Arc::default())
}

fn submit_to<I>(
    queue: &wgpu::Queue,
    label: &'static str,
    commands: I,
    signal: Arc<Signal>,
) -> Submission
where
    I: IntoIterator<Item = wgpu::CommandBuffer>,
{
    /// Keeps another [`submit`] from landing between a submission and its callback.
    static BIND: Mutex<()> = Mutex::new(());
    with_stats(label, |entry| {
        entry.count += 1;
        entry.in_flight += 1;
    });
    let notify = Notify {
        signal: Arc::clone(&signal),
        timed: Some((label, Instant::now())),
    };
    let _bind = BIND.lock().unwrap_or_else(PoisonError::into_inner);
    #[allow(
        clippy::disallowed_methods,
        reason = "the one submit of the cluster: its completion callback is registered below"
    )]
    queue.submit(commands);
    #[cfg(test)]
    if PANIC_AFTER_SUBMIT.take() {
        panic!("injected panic after actual queue submission");
    }
    queue.on_submitted_work_done(move || {
        let mut notify = notify;
        notify.finish(Ok(()));
    });
    Submission {
        signal,
        validation: None,
    }
}

/// Block the calling thread until `submission` completed (see the module docs: no blocking poll).
pub fn wait(device: &wgpu::Device, submission: &Submission) -> Result<(), WaitError> {
    block(device, &submission.signal, submission.validation.as_deref())
}

/// [`wait`] for everything submitted to `queue` so far (an empty submission marks it): for a caller
/// that needs the device quiet (tests, teardown, an error handler that must have run) rather than
/// one submission of its own.
pub fn wait_idle(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), WaitError> {
    wait(device, &submit(queue, "gpu-info wait_idle", []))
}

/// Map `slice` for reading and block the calling thread until the map finished: the buffer's last
/// submitted use (normally the copy into it) has completed by then. On `Ok` the caller reads
/// `slice.get_mapped_range()` and unmaps the buffer; on `Err` nothing is mapped.
pub fn map_read(device: &wgpu::Device, slice: &wgpu::BufferSlice<'_>) -> Result<(), WaitError> {
    let signal: Arc<Signal> = Arc::default();
    let notify = Notify {
        signal: Arc::clone(&signal),
        timed: None,
    };
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let mut notify = notify;
        notify.finish(result.map_err(WaitError::Map));
    });
    block(device, &signal, None)
}

/// Drive a wgpu future (an error-scope pop, an adapter request) to completion on the calling
/// thread with `pollster` (a mutex and condition variable; see the module docs).
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

/// A completion outcome, `None` until its callback ran or was dropped uncalled, and the condition
/// variable its waiter sleeps on.
type Signal = (Mutex<Option<Result<(), WaitError>>>, Condvar);

/// Owned by a completion callback: records its outcome, or [`WaitError::Dropped`] when wgpu drops
/// the callback uncalled.
struct Notify {
    signal: Arc<Signal>,
    /// A submission's label and submit time, counted in [`submit_stats`] at its first outcome; `None`
    /// for a map callback.
    timed: Option<(&'static str, Instant)>,
}

impl Notify {
    /// Record the first outcome (and a submission's time) and wake the waiter.
    fn finish(&mut self, result: Result<(), WaitError>) {
        if let Some((label, submitted)) = self.timed.take() {
            let took = submitted.elapsed();
            with_stats(label, |entry| {
                entry.in_flight = entry.in_flight.saturating_sub(1);
                entry.total += took;
                entry.max = entry.max.max(took);
            });
            if took > SLOW_SUBMISSION {
                log::debug!(
                    "gpu-info: submission '{label}' took {took:?} from submit to completion"
                );
            }
        }
        let (outcome, changed) = &*self.signal;
        let mut outcome = outcome.lock().unwrap_or_else(PoisonError::into_inner);
        if outcome.is_none() {
            *outcome = Some(result);
        }
        changed.notify_all();
    }
}

impl Drop for Notify {
    fn drop(&mut self) {
        self.finish(Err(WaitError::Dropped));
    }
}

/// Poll `device` without blocking until `signal` holds an outcome, sleeping on its condition
/// variable for at most [`POLL_PERIOD`] between polls; observers retain the outcome.
fn block(
    device: &wgpu::Device,
    signal: &Signal,
    validation: Option<&Validation>,
) -> Result<(), WaitError> {
    let (outcome, changed) = signal;
    loop {
        if let Some(result) = self::outcome(signal, validation) {
            return result;
        }
        poll(device)?;
        let guard = outcome.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.is_none() || validation.is_some() {
            let _slept = changed
                .wait_timeout(guard, POLL_PERIOD)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

fn submit_checked<I>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    commands: I,
    signal: Arc<Signal>,
) -> Checked<Submission>
where
    I: IntoIterator<Item = wgpu::CommandBuffer>,
{
    checked(device, || submit_to(queue, label, commands, signal))
}

#[cfg(test)]
thread_local! {
    static PANIC_AFTER_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PANIC_ON_POLL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn poll(device: &wgpu::Device) -> Result<wgpu::PollStatus, WaitError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(test)]
        if PANIC_ON_POLL.take() {
            panic!("injected fatal device poll");
        }
        #[allow(
            clippy::disallowed_methods,
            reason = "PollType::Poll never blocks and has no index to assert on (module docs)"
        )]
        device.poll(wgpu::PollType::Poll)
    }))
    .map_err(|payload| WaitError::PollPanicked(panic_message(payload.as_ref())))?
    .map_err(WaitError::Poll)
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn unscoped_submission_does_not_claim_validation() {
        let submission = Submission::pending();
        assert!(submission.validation_status().is_none());
        *submission.signal.0.lock().expect("signal") = Some(Ok(()));
        assert!(matches!(submission.status(), Some(Ok(()))));
        assert!(submission.validation_status().is_none());
    }

    /// A callback wgpu drops without calling (device loss) wakes the waiter with `Dropped`
    /// instead of blocking it forever.
    #[test]
    fn dropped_callback_reports_dropped() {
        let signal: Arc<Signal> = Arc::default();
        drop(Notify {
            signal: Arc::clone(&signal),
            timed: None,
        });
        let outcome = signal.0.lock().expect("signal").take();
        assert!(matches!(outcome, Some(Err(WaitError::Dropped))));
    }

    /// A real copy through `map_read` returns the bytes the queue wrote.
    #[test]
    #[ignore = "requires GPU"]
    fn map_read_returns_written_bytes() {
        let gpu = crate::shared_device().expect("shared device");
        let bytes: Vec<u8> = (0..=255).collect();
        let size = bytes.len() as u64;
        let source = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback test source"),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let target = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback test target"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&source, 0, &bytes);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&source, 0, &target, 0, size);
        let _copy = submit(&gpu.queue, "gpu-info test", [encoder.finish()]);
        map_read(&gpu.device, &target.slice(..)).expect("map");
        let mapped = target.slice(..).get_mapped_range().expect("range");
        assert_eq!(&mapped[..], &bytes[..]);
        drop(mapped);
        target.unmap();
    }

    /// Spin `rounds` LCG iterations per invocation over `data`: GPU work long enough to be measured,
    /// with a result the CPU can check ([`lcg`]).
    const BUSY: &str = "
        @group(0) @binding(0) var<storage, read_write> data: array<u32>;
        struct Rounds { n: u32 }
        @group(0) @binding(1) var<uniform> rounds: Rounds;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            var x = data[id.x];
            for (var i = 0u; i < rounds.n; i++) { x = x * 1664525u + 1013904223u; }
            data[id.x] = x;
        }";

    /// The CPU oracle of [`BUSY`]: `rounds` LCG steps from `seed`.
    fn lcg(seed: u32, rounds: u32) -> u32 {
        (0..rounds).fold(seed, |x, _| {
            x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
        })
    }

    /// Invocations of the tests' busy dispatches: 4096 workgroups of 64.
    const INVOCATIONS: u32 = 64 * 4096;

    /// [`BUSY`] ready to dispatch: its pipeline, a data buffer and the rounds uniform, made once per
    /// thread so a run costs one submission, not a shader compile.
    struct Busy {
        invocations: u32,
        pipeline: wgpu::ComputePipeline,
        data: wgpu::Buffer,
        uniform: wgpu::Buffer,
        group: wgpu::BindGroup,
    }

    impl Busy {
        fn new(device: &wgpu::Device) -> Self {
            Self::with(device, INVOCATIONS)
        }

        /// [`Busy`] of `invocations` (a multiple of 64).
        fn with(device: &wgpu::Device, invocations: u32) -> Self {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wait test busy"),
                source: wgpu::ShaderSource::Wgsl(BUSY.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("wait test busy"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let data = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wait test data"),
                size: u64::from(invocations) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wait test rounds"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            Self {
                invocations,
                pipeline,
                data,
                uniform,
                group,
            }
        }

        /// Seed the data with each invocation's index and run `rounds` LCG steps on it.
        fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue, rounds: u32) -> Submission {
            let seeds: Vec<u8> = (0..self.invocations).flat_map(u32::to_le_bytes).collect();
            queue.write_buffer(&self.data, 0, &seeds);
            queue.write_buffer(
                &self.uniform,
                0,
                &[rounds, 0, 0, 0].map(u32::to_le_bytes).concat(),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.group, &[]);
                pass.dispatch_workgroups(self.invocations / 64, 1, 1);
            }
            submit(queue, "gpu-info test busy", [encoder.finish()])
        }
    }

    /// The first `count` words of `data` (copied out and mapped after `wait` returned).
    fn words(gpu: &crate::SharedGpu, data: &wgpu::Buffer, count: u32) -> Vec<u32> {
        let size = u64::from(count) * 4;
        let target = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wait test words"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(data, 0, &target, 0, size);
        let _copy = submit(&gpu.queue, "gpu-info test", [encoder.finish()]);
        map_read(&gpu.device, &target.slice(..)).expect("map");
        let mapped = target.slice(..).get_mapped_range().expect("range");
        let (words, remainder) = mapped.as_chunks::<4>();
        assert!(
            remainder.is_empty(),
            "mapped word buffer is not u32-aligned"
        );
        let out = words.iter().copied().map(u32::from_le_bytes).collect();
        drop(mapped);
        target.unmap();
        out
    }

    /// Rounds of busy work that take at least `at_least` on this GPU, and how long they took
    /// (submit to completion).
    fn calibrate(gpu: &crate::SharedGpu, at_least: Duration) -> (u32, Duration) {
        let busy = Busy::new(&gpu.device);
        let mut rounds = 1u32 << 4;
        loop {
            let started = Instant::now();
            wait(&gpu.device, &busy.run(&gpu.device, &gpu.queue, rounds))
                .expect("calibration wait");
            let took = started.elapsed();
            if took >= at_least {
                return (rounds, took);
            }
            assert!(
                rounds < 1 << 30,
                "the GPU finished {rounds} rounds in {took:?}"
            );
            rounds = rounds.saturating_mul(4);
        }
    }

    /// Every submission is counted under its label: the count, nothing left in flight after the
    /// wait, and a time at least as long as the work (>= 50 ms here). RED when `submit` records nothing
    /// or `finish` does not record the time.
    #[test]
    #[ignore = "requires GPU"]
    fn submissions_are_counted_under_their_label() {
        let gpu = crate::shared_device().expect("shared device");
        let (rounds, work) = calibrate(gpu, Duration::from_millis(50));
        let busy = Busy::new(&gpu.device);
        for _ in 0..2 {
            wait(&gpu.device, &busy.run(&gpu.device, &gpu.queue, rounds)).expect("wait");
        }
        let stats = submit_stats();
        let entry = stats
            .iter()
            .find(|entry| entry.label == "gpu-info test busy")
            .expect("the label is counted");
        assert!(entry.count >= 2, "{entry:?}");
        assert_eq!(entry.in_flight, 0, "{entry:?}");
        assert!(entry.max * 2 >= work, "{entry:?} for {work:?} of work");
        assert!(entry.total >= entry.max, "{entry:?}");
    }

    /// `wait` returns only after the work completed: measured from `submit`'s return, it lasts at
    /// least half the calibrated work (>= 300 ms), and the results equal the CPU oracle. RED if
    /// `wait` returned early (a callback bound to nothing, or fired before the work).
    #[test]
    #[ignore = "requires GPU"]
    fn wait_returns_after_the_work_completed() {
        let gpu = crate::shared_device().expect("shared device");
        let (rounds, work) = calibrate(gpu, Duration::from_millis(300));
        let busy = Busy::new(&gpu.device);
        let submission = busy.run(&gpu.device, &gpu.queue, rounds);
        let started = Instant::now();
        wait(&gpu.device, &submission).expect("wait");
        let waited = started.elapsed();
        assert!(
            waited * 2 >= work,
            "wait returned after {waited:?}; the work takes {work:?}"
        );
        let got = words(gpu, &busy.data, 64);
        let want: Vec<u32> = (0..64).map(|seed| lcg(seed, rounds)).collect();
        assert_eq!(got, want, "the waited-for work's results");
    }

    /// Long work on [`crate::compute_device`] does not hold small work on [`crate::shared_device`]
    /// behind it when its workgroups are short (16384 groups of ~400 ms of work, as the fractal's
    /// direct render dispatches): the OS time-slices the two devices at workgroup boundaries
    /// (measured 12 ms median against 400 ms on one device). RED when `compute_device` hands out the
    /// shared device: the probe waits for the whole dispatch.
    #[test]
    #[ignore = "requires GPU"]
    fn compute_device_work_does_not_hold_the_shared_device() {
        let shared = crate::shared_device().expect("shared device");
        let compute = crate::compute_device().expect("compute device");
        let heavy = Busy::with(&compute.device, 64 * 16384);
        let mut rounds = 1u32 << 4;
        let busy = loop {
            let started = Instant::now();
            wait(
                &compute.device,
                &heavy.run(&compute.device, &compute.queue, rounds),
            )
            .expect("wait");
            let took = started.elapsed();
            if took >= Duration::from_millis(400) {
                break took;
            }
            assert!(
                rounds < 1 << 30,
                "the GPU finished {rounds} rounds in {took:?}"
            );
            rounds = rounds.saturating_mul(2);
        };
        // A probe of 64 invocations: its own cost is negligible, so its latency is the queueing.
        let probe = Busy::with(&shared.device, 64);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runner = std::thread::spawn({
            let stop = Arc::clone(&stop);
            move || {
                // Two dispatches in flight, so the compute queue never drains between them.
                let mut runs = 0u32;
                let mut in_flight = heavy.run(&compute.device, &compute.queue, rounds);
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let next = heavy.run(&compute.device, &compute.queue, rounds);
                    wait(&compute.device, &in_flight).expect("heavy wait");
                    in_flight = next;
                    runs += 1;
                }
                wait(&compute.device, &in_flight).expect("heavy wait");
                runs
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        let mut latencies = Vec::new();
        let started = Instant::now();
        while started.elapsed() < busy * 3 {
            let probed = Instant::now();
            wait(&shared.device, &probe.run(&shared.device, &shared.queue, 1)).expect("probe wait");
            latencies.push(probed.elapsed());
            std::thread::sleep(Duration::from_millis(20));
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let runs = runner.join().expect("runner");
        latencies.sort();
        let median = latencies[latencies.len() / 2];
        assert!(
            runs >= 2,
            "only {runs} heavy dispatches ran beside the probes"
        );
        assert!(
            median < Duration::from_millis(50) && median * 8 < busy,
            "small work on the shared device waited {median:?} (median of {}) behind {busy:?} \
             dispatches on the compute device",
            latencies.len()
        );
    }

    /// While a thread waits for long GPU work, another thread's resource release (`Buffer::destroy`
    /// takes the device's snatch lock for writing, as `Surface::present` does) is not held until the
    /// GPU finishes. RED with a blocking `PollType::Wait` in `block`: the releases wait for the
    /// whole GPU work.
    #[test]
    #[ignore = "requires GPU"]
    fn a_waiter_does_not_hold_a_present_behind_the_gpu() {
        let gpu = crate::shared_device().expect("shared device");
        let (rounds, busy) = calibrate(gpu, Duration::from_millis(400));
        let work = Busy::new(&gpu.device);
        let submission = work.run(&gpu.device, &gpu.queue, rounds);
        let started = Instant::now();
        let waiter = std::thread::spawn({
            let device = gpu.device.clone();
            move || wait(&device, &submission).map(|()| Instant::now())
        });
        let mut slowest = Duration::ZERO;
        let mut releases = 0;
        while !waiter.is_finished() && started.elapsed() < busy / 2 {
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wait test release"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let release = Instant::now();
            buffer.destroy();
            slowest = slowest.max(release.elapsed());
            releases += 1;
            std::thread::sleep(Duration::from_millis(5));
        }
        let finished = waiter.join().expect("waiter").expect("wait");
        assert!(
            finished.duration_since(started) >= busy / 2,
            "the GPU work ended before the releases were measured"
        );
        assert!(
            slowest < Duration::from_millis(30),
            "a release waited {slowest:?} behind a waiter ({busy:?} of GPU work)"
        );
        assert!(
            releases >= 10,
            "only {releases} releases during the GPU work"
        );
    }
}
