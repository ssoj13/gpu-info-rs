//! Context-local, charged resource reuse. Logical bytes exclude driver overhead.
//!
//! Keep tracked owners in every binding/view/recording until proven retirement.
//! Failed work must taint its owners before releasing them. Cloning a native wgpu
//! handle through Deref is untracked interop and requires retaining its owner.

use crate::budget::{AdmissionError, Charged, MemoryBudget};
use crate::{Checked, Pooled, ResourcePool, WaitError, checked};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::task::Poll;

/// All properties required to reuse an unmapped buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
}

/// Full allocation descriptor. Only the compatible sRGB alternate view is allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSpec {
    pub size: wgpu::Extent3d,
    pub dimension: wgpu::TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub view_format: Option<wgpu::TextureFormat>,
}

/// A stack slice can describe the complete operation before any GPU allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spec {
    Buffer(BufferSpec),
    Texture(TextureSpec),
}

/// Admission, descriptor and GPU failures remain distinguishable.
#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    #[error(transparent)]
    Admission(#[from] AdmissionError),
    #[error("invalid GPU operation input: {0}")]
    Input(String),
    #[error("invalid GPU resource descriptor: {0}")]
    InvalidDescriptor(&'static str),
    #[error("logical allocation size is unknown for {0:?}")]
    UnsupportedFormat(wgpu::TextureFormat),
    #[error("logical allocation byte count overflow")]
    Overflow,
    #[error("resource metadata allocation failed")]
    HostAllocation,
    #[error(transparent)]
    Gpu(#[from] WaitError),
    #[error("resource slot is absent or has a different kind")]
    Slot,
    #[error("upload needs an exclusively owned mapped staging buffer")]
    UploadState,
    #[error("mapped staging access failed")]
    MapRange(#[source] wgpu::MapRangeError),
}

impl BufferSpec {
    fn staging(self) -> bool {
        self.usage == (wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC)
    }

    pub fn bytes(self, limits: &wgpu::Limits) -> Result<u64, ResourceError> {
        if self.size == 0
            || self.size > limits.max_buffer_size
            || !self.size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            || self.usage.is_empty()
            || (self.usage.bits() & !wgpu::BufferUsages::all().bits()) != 0
        {
            return Err(ResourceError::InvalidDescriptor("buffer size/usage"));
        }
        if self.usage.contains(wgpu::BufferUsages::MAP_WRITE) && !self.staging() {
            return Err(ResourceError::InvalidDescriptor(
                "MAP_WRITE requires COPY_SRC staging",
            ));
        }
        if self.usage.contains(wgpu::BufferUsages::MAP_READ)
            && self.usage != (wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)
        {
            return Err(ResourceError::InvalidDescriptor(
                "MAP_READ requires COPY_DST",
            ));
        }
        if self.staging() && !self.size.is_multiple_of(wgpu::MAP_ALIGNMENT) {
            return Err(ResourceError::InvalidDescriptor("staging map alignment"));
        }
        Ok(self.size)
    }
}

impl TextureSpec {
    pub fn bytes(self, limits: &wgpu::Limits) -> Result<u64, ResourceError> {
        let size = self.size;
        if size.width == 0
            || size.height == 0
            || size.depth_or_array_layers == 0
            || self.usage.is_empty()
            || (self.usage.bits() & !wgpu::TextureUsages::all().bits()) != 0
            || self.mip_level_count == 0
            || self.mip_level_count > size.max_mips(self.dimension)
            || !matches!(self.sample_count, 1 | 2 | 4 | 8 | 16)
        {
            return Err(ResourceError::InvalidDescriptor(
                "texture extent/mips/samples/usage",
            ));
        }
        let extent_ok = match self.dimension {
            wgpu::TextureDimension::D1 => {
                size.width <= limits.max_texture_dimension_1d
                    && size.height == 1
                    && size.depth_or_array_layers == 1
                    && self.sample_count == 1
            }
            wgpu::TextureDimension::D2 => {
                size.width <= limits.max_texture_dimension_2d
                    && size.height <= limits.max_texture_dimension_2d
                    && size.depth_or_array_layers <= limits.max_texture_array_layers
            }
            wgpu::TextureDimension::D3 => {
                size.width <= limits.max_texture_dimension_3d
                    && size.height <= limits.max_texture_dimension_3d
                    && size.depth_or_array_layers <= limits.max_texture_dimension_3d
                    && self.sample_count == 1
            }
        };
        if !extent_ok
            || (self.sample_count > 1
                && (self.mip_level_count != 1 || size.depth_or_array_layers != 1))
        {
            return Err(ResourceError::InvalidDescriptor(
                "texture dimensions exceed enabled limits",
            ));
        }
        if self
            .view_format
            .is_some_and(|view| view.remove_srgb_suffix() != self.format.remove_srgb_suffix())
        {
            return Err(ResourceError::InvalidDescriptor(
                "incompatible texture view format",
            ));
        }
        let block = self
            .format
            .block_copy_size(None)
            .ok_or(ResourceError::UnsupportedFormat(self.format))?;
        let (bw, bh) = self.format.block_dimensions();
        if !size.width.is_multiple_of(bw) || !size.height.is_multiple_of(bh) {
            return Err(ResourceError::InvalidDescriptor(
                "compressed texture block alignment",
            ));
        }
        let mut bytes = 0u64;
        for level in 0..self.mip_level_count {
            let mip = size.mip_level_size(level, self.dimension);
            let level_bytes = u64::from(mip.width.div_ceil(bw))
                .checked_mul(u64::from(mip.height.div_ceil(bh)))
                .and_then(|n| n.checked_mul(u64::from(mip.depth_or_array_layers)))
                .and_then(|n| n.checked_mul(u64::from(self.sample_count)))
                .and_then(|n| n.checked_mul(u64::from(block)))
                .ok_or(ResourceError::Overflow)?;
            bytes = bytes
                .checked_add(level_bytes)
                .ok_or(ResourceError::Overflow)?;
        }
        Ok(bytes)
    }

    fn normalized(mut self) -> Self {
        if self.view_format == Some(self.format) {
            self.view_format = None;
        }
        self
    }
}

impl Spec {
    pub fn bytes(self, limits: &wgpu::Limits) -> Result<u64, ResourceError> {
        match self {
            Self::Buffer(spec) => spec.bytes(limits),
            Self::Texture(spec) => spec.bytes(limits),
        }
    }

    fn normalized(self) -> Self {
        match self {
            Self::Texture(spec) => Self::Texture(spec.normalized()),
            other => other,
        }
    }

    fn reusable(self) -> bool {
        !matches!(self, Self::Buffer(spec) if spec.usage.contains(wgpu::BufferUsages::MAP_READ))
    }
}

#[derive(Debug)]
enum Native {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture),
}

#[derive(Debug)]
struct Allocation {
    native: Native,
    spec: Spec,
    bytes: u64,
    mapped: AtomicBool,
}

impl Pooled for Allocation {
    type Key = Spec;
    fn key(&self) -> Spec {
        self.spec
    }
    fn bytes(&self) -> u64 {
        self.bytes
    }
}

struct Inner {
    device: Arc<wgpu::Device>,
    memory: MemoryBudget,
    loss: Arc<Mutex<Option<String>>>,
    pool: ResourcePool<Charged<Allocation>>,
    remaps: Mutex<Vec<Remap>>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resources")
            .field("memory", &self.memory.snapshot())
            .field("idle", &self.pool.idle_count())
            .finish_non_exhaustive()
    }
}

/// One resource owner per adopted context; clones share the same pool and charges.
#[derive(Clone, Debug)]
pub struct Resources(Arc<Inner>);

impl Resources {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        memory: MemoryBudget,
        loss: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self(Arc::new(Inner {
            device,
            memory,
            loss,
            // Idle retention is opt-in. Admission is independently host-configured.
            pool: ResourcePool::with_limits(0, 256),
            remaps: Mutex::new(Vec::new()),
        }))
    }

    pub fn set_idle_budget(&self, bytes: u64) {
        self.0.pool.set_budget(bytes);
    }
    pub fn idle(&self) -> (usize, u64) {
        self.0.pool.idle_count()
    }
    pub fn pending(&self) -> usize {
        self.0
            .remaps
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Check the entire descriptor plan, check out idle matches, then reserve all
    /// misses before any creation. No resources escape until all scopes accept.
    pub fn prepare(&self, specs: &[Spec]) -> Result<Prepared, ResourceError> {
        if let Some(reason) = self
            .0
            .loss
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
        {
            return Err(WaitError::DeviceLost(reason).into());
        }
        let limits = self.0.device.limits();
        for spec in specs {
            spec.bytes(&limits)?;
        }
        let mut batch = Batch {
            resources: self.clone(),
            slots: Vec::new(),
        };
        batch
            .slots
            .try_reserve_exact(specs.len())
            .map_err(|_| ResourceError::HostAllocation)?;
        let mut missing = 0u64;
        for &spec in specs {
            let spec = spec.normalized();
            let allocation = if spec.reusable() {
                self.0.pool.take(spec)
            } else {
                None
            };
            if allocation.is_none() {
                missing = missing
                    .checked_add(spec.bytes(&limits)?)
                    .ok_or(ResourceError::Overflow)?;
            }
            batch
                .slots
                .push(allocation.map(|allocation| Owned::new(allocation, &self.0, true)));
        }
        let mut reservation = match self.0.memory.reserve(missing) {
            Ok(reservation) => reservation,
            Err(AdmissionError::Capacity {
                requested,
                used,
                limit,
            }) => {
                // Checked-out matches are held by batch; only other idle capacity
                // is evicted, without changing the host's configured idle policy.
                self.0
                    .pool
                    .evict(used.saturating_add(requested).saturating_sub(limit));
                self.0.memory.reserve(missing)?
            }
            Err(error) => return Err(error.into()),
        };
        let resources = self.clone();
        let checked = checked(&self.0.device, move || -> Result<Batch, ResourceError> {
            for (slot, &spec) in batch.slots.iter_mut().zip(specs) {
                if slot.is_some() {
                    continue;
                }
                let spec = spec.normalized();
                let bytes = spec.bytes(&limits)?;
                let (native, mapped) = match spec {
                    Spec::Buffer(key) => (
                        Native::Buffer(resources.0.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("tracked buffer"),
                            size: key.size,
                            usage: key.usage,
                            mapped_at_creation: key.staging(),
                        })),
                        key.staging(),
                    ),
                    Spec::Texture(key) => (
                        Native::Texture(resources.0.device.create_texture(
                            &wgpu::TextureDescriptor {
                                label: Some("tracked texture"),
                                size: key.size,
                                dimension: key.dimension,
                                mip_level_count: key.mip_level_count,
                                sample_count: key.sample_count,
                                format: key.format,
                                usage: key.usage,
                                view_formats: key.view_format.as_slice(),
                            },
                        )),
                        false,
                    ),
                };
                let allocation = reservation.charge(Allocation {
                    native,
                    spec,
                    bytes,
                    mapped: AtomicBool::new(mapped),
                })?;
                *slot = Some(Owned::new(allocation, &resources.0, false));
            }
            Ok(batch)
        });
        Ok(Prepared { checked })
    }

    /// Harvest CPU-only map callbacks. The host's existing WorkQueue collection
    /// drives device polling; this method never waits or creates a helper thread.
    pub fn collect(&self) -> Result<usize, ResourceError> {
        let mut completed = Vec::new();
        {
            let mut remaps = self.0.remaps.lock().unwrap_or_else(PoisonError::into_inner);
            completed
                .try_reserve_exact(remaps.len())
                .map_err(|_| ResourceError::HostAllocation)?;
            let mut at = 0;
            while at < remaps.len() {
                if remaps[at]
                    .signal
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .is_none()
                {
                    at += 1;
                } else {
                    completed.push(remaps.swap_remove(at));
                }
            }
        }
        let count = completed.len();
        let lost = self
            .0
            .loss
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some();
        let mut failure = None;
        for remap in completed {
            let outcome = remap
                .signal
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            match outcome {
                Some(Ok(())) if !lost => {
                    remap
                        .allocation
                        .resource_untracked()
                        .mapped
                        .store(true, Ordering::Release);
                    self.0.pool.put(remap.allocation);
                }
                Some(Ok(())) => {}
                Some(Err(error)) => failure = Some(ResourceError::Gpu(error)),
                None => unreachable!("completed signal"),
            }
        }
        failure.map_or(Ok(count), Err)
    }
}

struct Owned {
    allocation: Option<Charged<Allocation>>,
    pool: Weak<Inner>,
    healthy: AtomicBool,
}

impl Owned {
    fn new(allocation: Charged<Allocation>, pool: &Arc<Inner>, healthy: bool) -> Arc<Self> {
        Arc::new(Self {
            allocation: Some(allocation),
            pool: Arc::downgrade(pool),
            healthy: AtomicBool::new(healthy),
        })
    }
    fn allocation(&self) -> &Allocation {
        self.allocation
            .as_ref()
            .expect("live allocation")
            .resource_untracked()
    }
    fn taint(&self) {
        self.healthy.store(false, Ordering::Release);
    }
}

impl std::fmt::Debug for Owned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracked")
            .field("allocation", &self.allocation)
            .finish_non_exhaustive()
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        let Some(allocation) = self.allocation.take() else {
            return;
        };
        if !self.healthy.load(Ordering::Acquire) || !allocation.key().reusable() {
            return;
        }
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        // An allocation that cannot be retained needs no staging remap.
        if allocation.bytes() > pool.pool.budget() || pool.pool.budget() == 0 {
            return;
        }
        if pool
            .loss
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
        {
            return;
        }
        let native = allocation.resource_untracked();
        if matches!(native.spec, Spec::Buffer(spec) if spec.staging())
            && !native.mapped.load(Ordering::Acquire)
        {
            let signal = Arc::new(Mutex::new(None));
            let mut remaps = pool.remaps.lock().unwrap_or_else(PoisonError::into_inner);
            if remaps.try_reserve(1).is_err() {
                return;
            }
            // Keep the allocation before starting its callback, but never encode,
            // poll or destroy a GPU object under the metadata lock.
            remaps.push(Remap {
                allocation,
                signal: signal.clone(),
            });
            let buffer = match &remaps
                .last()
                .expect("inserted remap")
                .allocation
                .resource_untracked()
                .native
            {
                Native::Buffer(buffer) => buffer.clone(),
                Native::Texture(_) => unreachable!(),
            };
            drop(remaps);
            let mut notify = MapNotify(signal);
            // A malformed untracked alias may make the backend unwind. Drop must
            // remain non-panicking; callback loss is harvested as a typed failure.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                buffer.map_async(wgpu::MapMode::Write, .., move |result| {
                    notify.finish(result.map_err(WaitError::from));
                });
            }));
        } else {
            pool.pool.put(allocation);
        }
    }
}

struct Remap {
    allocation: Charged<Allocation>,
    signal: Arc<Mutex<Option<Result<(), WaitError>>>>,
}

struct MapNotify(Arc<Mutex<Option<Result<(), WaitError>>>>);
impl MapNotify {
    fn finish(&mut self, result: Result<(), WaitError>) {
        let mut outcome = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if outcome.is_none() {
            *outcome = Some(result);
        }
    }
}
impl Drop for MapNotify {
    fn drop(&mut self) {
        self.finish(Err(WaitError::Dropped));
    }
}

/// Shared tracked buffer. Bindings and GPU work must retain this owner.
#[derive(Clone, Debug)]
pub struct Buffer(Arc<Owned>);
impl Deref for Buffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        match &self.0.allocation().native {
            Native::Buffer(buffer) => buffer,
            _ => unreachable!(),
        }
    }
}
impl Buffer {
    /// Permanently exclude failed/indeterminate work's allocation from reuse.
    pub fn taint(&self) {
        self.0.taint();
    }
    pub fn bytes(&self) -> u64 {
        self.0.allocation().bytes
    }
    pub fn belongs_to(&self, resources: &Resources) -> bool {
        self.0.pool.ptr_eq(&Arc::downgrade(&resources.0))
    }
    pub fn spec(&self) -> BufferSpec {
        match self.0.allocation().spec {
            Spec::Buffer(spec) => spec,
            _ => unreachable!(),
        }
    }
    /// Fill an exclusive staging buffer, then unmap it before GPU encoding.
    /// The destination copy must retain this owner through GPU retirement.
    pub fn upload(&mut self, offset: u64, bytes: &[u8]) -> Result<(), ResourceError> {
        if Arc::strong_count(&self.0) != 1
            || !self.spec().staging()
            || !self.0.allocation().mapped.load(Ordering::Acquire)
            || !offset.is_multiple_of(wgpu::MAP_ALIGNMENT)
            || bytes.is_empty()
            || !(bytes.len() as u64).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            || offset
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > self.size())
        {
            return Err(ResourceError::UploadState);
        }
        {
            let mut view = self
                .get_mapped_range_mut(offset..offset + bytes.len() as u64)
                .map_err(ResourceError::MapRange)?;
            view.copy_from_slice(bytes);
        }
        self.unmap();
        self.0.allocation().mapped.store(false, Ordering::Release);
        Ok(())
    }
}

/// Shared tracked texture. Native views must retain this same allocation owner.
#[derive(Clone, Debug)]
pub struct Texture(Arc<Owned>);
impl Deref for Texture {
    type Target = wgpu::Texture;
    fn deref(&self) -> &Self::Target {
        match &self.0.allocation().native {
            Native::Texture(texture) => texture,
            _ => unreachable!(),
        }
    }
}
impl Texture {
    pub fn taint(&self) {
        self.0.taint();
    }
    pub fn bytes(&self) -> u64 {
        self.0.allocation().bytes
    }
    pub fn belongs_to(&self, resources: &Resources) -> bool {
        self.0.pool.ptr_eq(&Arc::downgrade(&resources.0))
    }
    pub fn spec(&self) -> TextureSpec {
        match self.0.allocation().spec {
            Spec::Texture(spec) => spec,
            _ => unreachable!(),
        }
    }
}

/// Unsubmitted creation and scope futures. Drop aborts without publishing resources.
#[derive(Debug)]
#[must_use]
pub struct Prepared {
    checked: Checked<Result<Batch, ResourceError>>,
}
impl Prepared {
    pub fn poll(&mut self) -> Poll<Result<Batch, ResourceError>> {
        match self.checked.poll() {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => Poll::Ready(
                result
                    .map_err(ResourceError::from)
                    .and_then(|batch| batch)
                    .map(Batch::accepted),
            ),
        }
    }
    pub async fn resolve(self) -> Result<Batch, ResourceError> {
        Ok(self.checked.resolve().await??.accepted())
    }
}

/// An accepted complete operation. Each slot is transferred once.
#[derive(Debug)]
pub struct Batch {
    resources: Resources,
    slots: Vec<Option<Arc<Owned>>>,
}
impl Batch {
    fn accepted(self) -> Self {
        for owner in self.slots.iter().flatten() {
            owner.healthy.store(true, Ordering::Release);
        }
        self
    }
    pub fn buffer(&mut self, index: usize) -> Result<Buffer, ResourceError> {
        let slot = self.slots.get_mut(index).ok_or(ResourceError::Slot)?;
        if !matches!(
            slot.as_ref().map(|owner| owner.allocation().spec),
            Some(Spec::Buffer(_))
        ) {
            return Err(ResourceError::Slot);
        }
        Ok(Buffer(slot.take().ok_or(ResourceError::Slot)?))
    }
    pub fn texture(&mut self, index: usize) -> Result<Texture, ResourceError> {
        let slot = self.slots.get_mut(index).ok_or(ResourceError::Slot)?;
        if !matches!(
            slot.as_ref().map(|owner| owner.allocation().spec),
            Some(Spec::Texture(_))
        ) {
            return Err(ResourceError::Slot);
        }
        Ok(Texture(slot.take().ok_or(ResourceError::Slot)?))
    }
    pub fn resources(&self) -> &Resources {
        &self.resources
    }
}

#[cfg(test)]
mod tests;
