//! The ONE blocking GPU wait of the cluster: wait for a submission ([`wait`]), or map a buffer slice
//! for reading and wait for the map ([`map_read`]), in slices of [`WAIT_SLICE`].
//!
//! **Why slices, never `PollType::wait_indefinitely`:** wgpu-core 30 `Device::maintain` holds the
//! device's snatch lock for reading while it waits on the fence (`device/resource.rs:797-879`), and
//! `Surface::present` (`present.rs:340`), `Buffer::destroy` / `unmap` and every other resource
//! release take it for writing. One thread waiting indefinitely for its readback therefore froze the
//! UI's `present` for as long as the GPU worked - Playa measured `Queue::present` at 1.8 s behind
//! background composites (ofx-rs plan0.md 3.H). A wait of at most [`WAIT_SLICE`] releases the lock
//! between slices, and `parking_lot`'s task-fair `RwLock` lets a waiting writer in before the next
//! slice takes it again, so a present waits at most one slice. A submission index is waited for,
//! never "the last submission": that is every thread's work, not the caller's.
//!
//! **Why here:** every wgpu consumer (ofx-rs `ofx::gpu_wgpu`, `ofx-fractal`, `ofx-host-wgpu`,
//! Playa's compositor, [`crate::GpuImage`]) waits on the same shared device, so this crate owns the
//! wait.
//!
//! **Why a mutex and condition variable, not a channel or a park executor:** plug-ins call this
//! on host threads. A blocking `std::sync::mpsc` receive and `std::thread::park` both reach
//! `std::thread::current()`, which on glibc registers a thread-exit destructor inside the calling
//! module that `dlclose` does not unregister (ofx-rs PLAN 2.5). [`block_on`] uses `pollster`,
//! which waits the same way.
//!
//! **Order for callers with error scopes:** call [`map_read`], then pop the scopes, then look at
//! the result: a scope that captured an out-of-memory error still names the real cause of a
//! failed map.

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

/// The longest one wait holds wgpu-core's device lock: the most a concurrent `present` or resource
/// release waits for a waiter (see the module docs).
pub const WAIT_SLICE: Duration = Duration::from_millis(1);

/// Why a readback wait failed. Every variant means the mapped range must not be read.
#[derive(Debug, thiserror::Error)]
pub enum ReadbackError {
    /// [`wgpu::Device::poll`] failed while waiting for the map (device lost, wrong index).
    #[error("device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    /// wgpu completed the map with an error.
    #[error("buffer map failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    /// wgpu dropped the map callback without calling it (the device or buffer is gone).
    #[error("buffer map callback dropped without running")]
    Dropped,
}

/// Block the calling thread until `submission` of `device` completed, and run the callbacks
/// (buffer maps, `on_submitted_work_done`) of every submission completed by then.
///
/// Waits in [`WAIT_SLICE`]s, so other threads' `present`, submissions and resource releases are
/// never held for longer than one slice (see the module docs). `Err` only for a device error or
/// an index the device never issued, never for the time the GPU takes.
pub fn wait(
    device: &wgpu::Device,
    submission: &wgpu::SubmissionIndex,
) -> Result<(), wgpu::PollError> {
    loop {
        match poll_slice(device, Some(submission)) {
            Err(wgpu::PollError::Timeout) => {}
            done => return done,
        }
    }
}

/// [`wait`] for everything submitted to `queue` so far (an empty submission marks it): for a caller
/// that needs the device quiet (tests, teardown, an error handler that must have run) rather than
/// one submission of its own.
pub fn wait_idle(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), wgpu::PollError> {
    wait(device, &queue.submit([]))
}

/// Map `slice` for reading and block the calling thread until the map finished.
///
/// Requests the map, then waits in [`WAIT_SLICE`]s until its callback ran (on this thread's poll
/// or another's): a copy into the buffer submitted before this call has completed by then. On `Ok`
/// the caller reads `slice.get_mapped_range()` and unmaps the buffer; on `Err` nothing is mapped.
pub fn map_read(device: &wgpu::Device, slice: &wgpu::BufferSlice<'_>) -> Result<(), ReadbackError> {
    let state: Arc<MapState> = Arc::new((Mutex::new(None), Condvar::new()));
    let notify = MapNotify(Arc::clone(&state));
    slice.map_async(wgpu::MapMode::Read, move |result| {
        notify.finish(Some(result))
    });
    // No index: the map completes with the last submission that used the buffer, which only wgpu
    // knows. Every slice processes what completed, so the loop ends as soon as the callback ran,
    // however much other work is queued behind it.
    while !recorded(&state) {
        match poll_slice(device, None) {
            Ok(()) => break,
            Err(wgpu::PollError::Timeout) => {}
            Err(error) => return Err(error.into()),
        }
    }
    take(&state)
}

/// Drive a wgpu future (an error-scope pop, an adapter request) to completion on the calling
/// thread with `pollster` (a mutex and condition variable; see the module docs).
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

/// One wait of at most [`WAIT_SLICE`] for `submission` (`None`: the device's last submission).
fn poll_slice(
    device: &wgpu::Device,
    submission: Option<&wgpu::SubmissionIndex>,
) -> Result<(), wgpu::PollError> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: submission.cloned(),
            timeout: Some(WAIT_SLICE),
        })
        .map(drop)
}

/// The map outcome: `None` until the callback ran or was dropped uncalled.
type MapState = (Mutex<Option<Result<(), ReadbackError>>>, Condvar);

/// Owned by the map callback: records its outcome, or [`ReadbackError::Dropped`] when wgpu drops
/// the callback uncalled.
struct MapNotify(Arc<MapState>);

impl MapNotify {
    /// Record the first outcome (`None` = dropped uncalled) and wake the waiter.
    fn finish(&self, result: Option<Result<(), wgpu::BufferAsyncError>>) {
        let (outcome, changed) = &*self.0;
        let mut outcome = outcome.lock().unwrap_or_else(PoisonError::into_inner);
        if outcome.is_none() {
            *outcome = Some(match result {
                Some(result) => result.map_err(ReadbackError::Map),
                None => Err(ReadbackError::Dropped),
            });
        }
        changed.notify_all();
    }
}

impl Drop for MapNotify {
    fn drop(&mut self) {
        self.finish(None);
    }
}

/// Whether the callback recorded an outcome yet.
fn recorded(state: &MapState) -> bool {
    state
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some()
}

/// Block until the callback recorded an outcome (a poll on another thread may still be running
/// it), then take it.
fn take(state: &MapState) -> Result<(), ReadbackError> {
    let (outcome, changed) = state;
    let mut outcome = outcome.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        if let Some(result) = outcome.take() {
            return result;
        }
        outcome = changed
            .wait(outcome)
            .unwrap_or_else(PoisonError::into_inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// A callback wgpu drops without calling (device loss) wakes the waiter with `Dropped`
    /// instead of blocking it forever.
    #[test]
    fn dropped_callback_reports_dropped() {
        let state: Arc<MapState> = Arc::new((Mutex::new(None), Condvar::new()));
        drop(MapNotify(Arc::clone(&state)));
        assert!(matches!(take(&state), Err(ReadbackError::Dropped)));
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
        gpu.queue.submit([encoder.finish()]);
        map_read(&gpu.device, &target.slice(..)).expect("map");
        let mapped = target.slice(..).get_mapped_range().expect("range");
        assert_eq!(&mapped[..], &bytes[..]);
        drop(mapped);
        target.unmap();
    }

    /// Spin `rounds` iterations per invocation over `buffer`: GPU work long enough to be measured.
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

    /// Submit busy work and return its index; `rounds` scales its duration.
    fn submit_busy(gpu: &crate::SharedGpu, rounds: u32) -> wgpu::SubmissionIndex {
        const INVOCATIONS: u32 = 64 * 4096;
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wait test busy"),
                source: wgpu::ShaderSource::Wgsl(BUSY.into()),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("wait test busy"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let data = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wait test data"),
            size: u64::from(INVOCATIONS) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wait test rounds"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &uniform,
            0,
            &[rounds, 0, 0, 0].map(u32::to_le_bytes).concat(),
        );
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(INVOCATIONS / 64, 1, 1);
        }
        gpu.queue.submit([encoder.finish()])
    }

    /// While a thread waits for long GPU work, another thread's resource release (`Buffer::destroy`
    /// takes the device's snatch lock for writing, as `Surface::present` does) is held at most
    /// about one slice, not until the GPU finishes. RED with an unsliced `wait_indefinitely` in
    /// `poll_slice`: the releases wait for the whole GPU work.
    #[test]
    #[ignore = "requires GPU"]
    fn a_waiter_does_not_hold_a_present_behind_the_gpu() {
        let gpu = crate::shared_device().expect("shared device");
        // Calibrate so the busy work runs for at least 400 ms on this GPU.
        let mut rounds = 1u32 << 12;
        let busy = loop {
            let started = Instant::now();
            let index = submit_busy(gpu, rounds);
            wait(&gpu.device, &index).expect("calibration wait");
            let took = started.elapsed();
            if took >= Duration::from_millis(400) {
                break took;
            }
            assert!(
                rounds < 1 << 30,
                "the GPU finished {rounds} rounds in {took:?}"
            );
            rounds = rounds.saturating_mul(4);
        };
        let index = submit_busy(gpu, rounds);
        let started = Instant::now();
        let waiter = std::thread::spawn({
            let device = gpu.device.clone();
            move || wait(&device, &index).map(|()| Instant::now())
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
