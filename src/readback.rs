//! The ONE blocking buffer-readback wait of the cluster: map a buffer slice for reading, poll the
//! device until the map has run, and report a failure as a typed [`ReadbackError`].
//!
//! **Why here:** every wgpu consumer (ofx-rs `ofx::gpu_wgpu`, `ofx-fractal`, [`crate::GpuImage`])
//! reads results back the same way, and each used to carry its own copy. This crate owns the
//! shared device, so it owns the wait too.
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

/// Why a readback wait failed. Every variant means the mapped range must not be read.
#[derive(Debug, thiserror::Error)]
pub enum ReadbackError {
    /// [`wgpu::Device::poll`] failed while waiting for the map (device lost, timeout).
    #[error("device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    /// wgpu completed the map with an error.
    #[error("buffer map failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    /// wgpu dropped the map callback without calling it (the device or buffer is gone).
    #[error("buffer map callback dropped without running")]
    Dropped,
}

/// Map `slice` for reading and block the calling thread until the map finished.
///
/// Requests the map, polls `device` with [`wgpu::PollType::wait_indefinitely`] (which also waits
/// for every submission, so a copy into the buffer submitted before this call has completed),
/// then waits for the map callback, which another thread's poll may be running. On `Ok` the
/// caller reads `slice.get_mapped_range()` and unmaps the buffer; on `Err` nothing is mapped.
pub fn map_read(device: &wgpu::Device, slice: &wgpu::BufferSlice<'_>) -> Result<(), ReadbackError> {
    let state: Arc<MapState> = Arc::new((Mutex::new(None), Condvar::new()));
    let notify = MapNotify(Arc::clone(&state));
    slice.map_async(wgpu::MapMode::Read, move |result| {
        notify.finish(Some(result))
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    wait(&state)
}

/// Drive a wgpu future (an error-scope pop, an adapter request) to completion on the calling
/// thread with `pollster` (a mutex and condition variable; see the module docs).
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
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

/// Block until the callback recorded an outcome, then take it.
fn wait(state: &MapState) -> Result<(), ReadbackError> {
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

    /// A callback wgpu drops without calling (device loss) wakes the waiter with `Dropped`
    /// instead of blocking it forever.
    #[test]
    fn dropped_callback_reports_dropped() {
        let state: Arc<MapState> = Arc::new((Mutex::new(None), Condvar::new()));
        drop(MapNotify(Arc::clone(&state)));
        assert!(matches!(wait(&state), Err(ReadbackError::Dropped)));
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
}
