//! Reused GPU resources: one bounded pool per device and kind, instead of a new buffer or texture per frame.
//!
//! ```text
//! take(key)            the most recently returned idle resource of `key`, else None (the caller creates one)
//! put(resource)        back into the pool; the least recently returned ones are evicted beyond the budget,
//!                      and a resource larger than the whole budget is dropped
//! take_buffer / take_texture   take, else create with the key's descriptor
//! ```
//!
//! **Why:** creating and dropping large resources every frame reaches the driver as `vkAllocateMemory` /
//! `vkFreeMemory` (a 4K Rgba32Float image is a dedicated allocation; small ones churn the allocator's 128/256 MB
//! blocks). On Windows those calls stall the whole process's GPU scheduling. MEASURED 2026-09-29 (Nsight Systems
//! 2026.1, Playa playback, RTX 3080 Ti): 64 `vkFreeMemory` calls took 4.8 s (up to 668 ms each) and 300
//! `vkAllocateMemory` up to 353 ms, while the UI's `vkQueuePresentKHR` waited up to 379 ms and the video decoder's
//! `vkQueueSubmit` up to 701 ms; wgpu-core also frees dropped resources inside a later `Queue::submit`'s `maintain`,
//! holding the device (741 ms seen). A pooled resource is allocated once.
//!
//! **Rules:** a pool holds resources of ONE device (a pool per device, or a `static` for a process-wide device such
//! as [`crate::compute_device`]). Only a resource its owner knows to be unshared may be put back: a `wgpu` handle can
//! be cloned, and a pooled resource still used elsewhere would be overwritten by its next taker. Reuse on the same
//! queue needs no wait (wgpu orders later writes after earlier reads); a taker must not assume any contents (clear
//! it when the work reads it before writing, e.g. an accumulator). The lock is held only to pick or return.
//!
//! **Where used:** `ofx-fractal` (chunk outputs, readbacks, palettes on the compute device), Playa `render_gpu`
//! (upload staging slots, readback staging, plate upload targets on the shared device).

use std::sync::Mutex;

/// A resource a [`ResourcePool`] keeps, found again by its key and counted by its bytes.
pub trait Pooled {
    /// What a resource of the same key can be reused for.
    type Key: PartialEq + Copy;
    /// This resource's key.
    fn key(&self) -> Self::Key;
    /// The memory it holds (for the budget).
    fn bytes(&self) -> u64;
}

/// A buffer's key: size and usage.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferKey {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
}

#[cfg(feature = "wgpu")]
impl Pooled for wgpu::Buffer {
    type Key = BufferKey;
    fn key(&self) -> BufferKey {
        BufferKey {
            size: self.size(),
            usage: self.usage(),
        }
    }
    fn bytes(&self) -> u64 {
        self.size()
    }
}

/// A 2D texture's key (one mip, one sample, one layer): size, format and usage.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

#[cfg(feature = "wgpu")]
impl Pooled for wgpu::Texture {
    type Key = TextureKey;
    fn key(&self) -> TextureKey {
        TextureKey {
            width: self.width(),
            height: self.height(),
            format: self.format(),
            usage: self.usage(),
        }
    }
    fn bytes(&self) -> u64 {
        let texel = u64::from(self.format().block_copy_size(None).unwrap_or(16));
        u64::from(self.width()) * u64::from(self.height()) * texel
    }
}

/// Idle resources of one device, least recently returned first, at most `budget` bytes of them.
pub struct ResourcePool<R> {
    idle: Mutex<Idle<R>>,
    budget: u64,
}

struct Idle<R> {
    slots: Vec<R>,
    bytes: u64,
}

impl<R> std::fmt::Debug for ResourcePool<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcePool")
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl<R: Pooled> ResourcePool<R> {
    /// An empty pool keeping at most `budget` bytes of idle resources (`const`: a `static` pool of a process-wide
    /// device).
    pub const fn new(budget: u64) -> Self {
        Self {
            idle: Mutex::new(Idle {
                slots: Vec::new(),
                bytes: 0,
            }),
            budget,
        }
    }

    fn idle(&self) -> std::sync::MutexGuard<'_, Idle<R>> {
        self.idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The most recently returned idle resource of `key` (the likeliest to be ready again), if any.
    pub fn take(&self, key: R::Key) -> Option<R> {
        let mut idle = self.idle();
        let at = idle.slots.iter().rposition(|slot| slot.key() == key)?;
        let slot = idle.slots.remove(at);
        idle.bytes -= slot.bytes();
        Some(slot)
    }

    /// Return `resource`, which no one else holds: kept for the next take of its key, evicting the least recently
    /// returned ones beyond the budget; one larger than the whole budget is dropped.
    pub fn put(&self, resource: R) {
        let size = resource.bytes();
        if size > self.budget {
            return;
        }
        let mut idle = self.idle();
        while idle.bytes + size > self.budget && !idle.slots.is_empty() {
            let old = idle.slots.remove(0);
            idle.bytes -= old.bytes();
        }
        idle.bytes += size;
        idle.slots.push(resource);
    }

    /// Idle resources and their bytes.
    pub fn idle_count(&self) -> (usize, u64) {
        let idle = self.idle();
        (idle.slots.len(), idle.bytes)
    }
}

#[cfg(feature = "wgpu")]
impl ResourcePool<wgpu::Buffer> {
    /// A buffer of `key` from the pool, else a new unmapped one on `device` labelled `label`.
    pub fn take_buffer(&self, device: &wgpu::Device, key: BufferKey, label: &str) -> wgpu::Buffer {
        self.take(key).unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: key.size,
                usage: key.usage,
                mapped_at_creation: false,
            })
        })
    }
}

#[cfg(feature = "wgpu")]
impl ResourcePool<wgpu::Texture> {
    /// A texture of `key` from the pool, else a new 2D one (one mip, one sample) on `device` labelled `label`.
    pub fn take_texture(
        &self,
        device: &wgpu::Device,
        key: TextureKey,
        label: &str,
    ) -> wgpu::Texture {
        self.take(key).unwrap_or_else(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: key.width.max(1),
                    height: key.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: key.format,
                usage: key.usage,
                view_formats: &[],
            })
        })
    }
}

/// A bound on how many jobs use a pool at once: a counting semaphore whose permit a job takes BEFORE it takes any
/// pooled resource and holds until it has returned them all.
///
/// **Why:** a pool keeps a budget of idle resources, but lends as many as are asked for. With more jobs at once than
/// the budget holds (Playa: up to 18 render workers staging 4K plates into a pool of three), every extra job creates a
/// resource and every return beyond the budget frees one: the churn the pool exists to stop (Nsight Systems, Playa
/// playback: 34 upload staging buffers of 127 MB, 18 upload targets, 13 readbacks created in 25 s even so). A gate
/// sized to the budget makes the pool's resources enough for every job it admits, and bounds the GPU work those jobs
/// queue. The permit is taken once per job and never while holding pooled resources, so no job waits holding what
/// another needs.
pub struct Gate {
    free: Mutex<usize>,
    freed: std::sync::Condvar,
}

impl std::fmt::Debug for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gate").finish_non_exhaustive()
    }
}

/// A job's admission through a [`Gate`], given back when dropped.
#[must_use = "the job is admitted while the permit lives"]
pub struct Permit<'gate>(&'gate Gate);

impl Gate {
    /// A gate admitting `permits` jobs at once (at least one).
    pub const fn new(permits: usize) -> Self {
        Self {
            free: Mutex::new(if permits == 0 { 1 } else { permits }),
            freed: std::sync::Condvar::new(),
        }
    }

    /// Wait until a job may run, and admit it. Worker threads only: it blocks.
    pub fn enter(&self) -> Permit<'_> {
        self.wait_free();
        Permit(self)
    }

    /// Admit a worker job while checking cancellation outside the gate's lock.
    /// A cancelled waiter consumes no permit and does not stop active jobs.
    pub fn enter_until(&self, mut keep_waiting: impl FnMut() -> bool) -> Option<Permit<'_>> {
        loop {
            if !keep_waiting() {
                return None;
            }
            let mut free = self
                .free
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if *free != 0 {
                *free -= 1;
                return Some(Permit(self));
            }
            let waited = self
                .freed
                .wait_timeout(free, std::time::Duration::from_millis(10))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            drop(waited);
        }
    }

    /// [`Self::enter`] for a job whose permit outlives the borrow (kept in a value that is handed on, such as a
    /// pending readback): the permit holds the gate.
    pub fn enter_owned(gate: &std::sync::Arc<Self>) -> OwnedPermit {
        gate.wait_free();
        OwnedPermit(std::sync::Arc::clone(gate))
    }

    fn wait_free(&self) {
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *free == 0 {
            free = self
                .freed
                .wait(free)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *free -= 1;
    }

    fn leave(&self) {
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *free += 1;
        self.freed.notify_one();
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.leave();
    }
}

/// A [`Gate`] admission that holds its gate ([`Gate::enter_owned`]), given back when dropped.
#[must_use = "the job is admitted while the permit lives"]
pub struct OwnedPermit(std::sync::Arc<Gate>);

impl std::fmt::Debug for OwnedPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedPermit").finish_non_exhaustive()
    }
}

impl Drop for OwnedPermit {
    fn drop(&mut self) {
        self.0.leave();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resource with a key and a size, no GPU.
    #[derive(Debug, PartialEq)]
    struct Fake {
        key: u32,
        bytes: u64,
        id: u32,
    }

    impl Pooled for Fake {
        type Key = u32;
        fn key(&self) -> u32 {
            self.key
        }
        fn bytes(&self) -> u64 {
            self.bytes
        }
    }

    fn fake(key: u32, bytes: u64, id: u32) -> Fake {
        Fake { key, bytes, id }
    }

    /// A take finds only its key, the most recently returned first; nothing of another key.
    #[test]
    fn take_finds_the_latest_of_its_key() {
        let pool = ResourcePool::new(100);
        pool.put(fake(1, 10, 1));
        pool.put(fake(2, 10, 2));
        pool.put(fake(1, 10, 3));
        assert_eq!(pool.take(1).map(|f| f.id), Some(3));
        assert_eq!(pool.take(1).map(|f| f.id), Some(1));
        assert_eq!(pool.take(1), None);
        assert_eq!(pool.idle_count(), (1, 10));
    }

    #[test]
    fn a_cancelled_gate_waiter_consumes_no_permit() {
        let gate = Gate::new(1);
        let owner = gate.enter();
        let mut polls = 0;
        assert!(
            gate.enter_until(|| {
                polls += 1;
                polls < 3
            })
            .is_none()
        );
        assert!(gate.enter_until(|| false).is_none());
        drop(owner);
        let admitted = gate.enter_until(|| true).expect("permit returned");
        drop(admitted);
        assert!(gate.enter_until(|| true).is_some());
    }

    /// A gate of two admits two jobs at once and a third only after one left: a waiting thread is
    /// released exactly by a permit's drop (ordered by channels, no timing).
    #[test]
    fn a_gate_admits_its_permits_and_waits_for_a_return() {
        use std::sync::mpsc::channel;
        let gate = Gate::new(2);
        let first = gate.enter();
        let second = gate.enter();
        let (entered, admitted) = channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _third = gate.enter();
                entered.send(()).expect("send");
            });
            assert!(
                admitted
                    .recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "a third job waits while two permits are out"
            );
            drop(first);
            admitted
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("admitted once a permit came back");
        });
        drop(second);
    }

    /// Beyond the budget the least recently returned are evicted; one larger than the budget is not kept.
    #[test]
    fn the_budget_evicts_the_oldest_and_refuses_the_oversized() {
        let pool = ResourcePool::new(30);
        pool.put(fake(1, 10, 1));
        pool.put(fake(1, 10, 2));
        pool.put(fake(1, 10, 3));
        pool.put(fake(2, 15, 4));
        // 10 + 10 + 10 + 15 > 30: ids 1 and 2 go (oldest first) until 10 + 15 fits.
        assert_eq!(pool.idle_count(), (2, 25));
        assert_eq!(pool.take(1).map(|f| f.id), Some(3));
        pool.put(fake(3, 31, 5));
        assert_eq!(pool.idle_count(), (1, 15));
    }
}
