use super::*;
use crate::{RecordedWork, WorkContext, WorkLease, WorkQueue};
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicUsize;

fn texture() -> TextureSpec {
    TextureSpec {
        size: wgpu::Extent3d {
            width: 8,
            height: 4,
            depth_or_array_layers: 1,
        },
        dimension: wgpu::TextureDimension::D2,
        mip_level_count: 1,
        sample_count: 1,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_format: None,
    }
}

#[test]
fn texture_bytes_include_mips_array_layers_and_multisampling() {
    let limits = wgpu::Limits::default();
    let mut key = texture();
    assert_eq!(key.bytes(&limits).unwrap(), 512);
    key.size.depth_or_array_layers = 3;
    key.mip_level_count = 4;
    assert_eq!(key.bytes(&limits).unwrap(), (32 + 8 + 2 + 1) * 3 * 16);
    key.size.depth_or_array_layers = 1;
    key.mip_level_count = 1;
    key.sample_count = 4;
    assert_eq!(key.bytes(&limits).unwrap(), 2048);
}

#[test]
fn texture_bytes_shrink_depth_only_for_three_dimensional_mips() {
    let mut key = texture();
    key.size = wgpu::Extent3d {
        width: 8,
        height: 4,
        depth_or_array_layers: 4,
    };
    key.dimension = wgpu::TextureDimension::D3;
    key.mip_level_count = 4;
    assert_eq!(
        key.bytes(&wgpu::Limits::default()).unwrap(),
        (128 + 16 + 2 + 1) * 16
    );
}

#[test]
fn compressed_mips_round_blocks_and_opaque_formats_are_rejected() {
    let mut key = texture();
    key.format = wgpu::TextureFormat::Bc1RgbaUnorm;
    key.mip_level_count = 4;
    assert_eq!(
        key.bytes(&wgpu::Limits::default()).unwrap(),
        (2 + 1 + 1 + 1) * 8
    );
    key.format = wgpu::TextureFormat::Depth24Plus;
    assert!(matches!(
        key.bytes(&wgpu::Limits::default()),
        Err(ResourceError::UnsupportedFormat(_))
    ));
    key.format = wgpu::TextureFormat::NV12;
    assert!(matches!(
        key.bytes(&wgpu::Limits::default()),
        Err(ResourceError::UnsupportedFormat(_))
    ));
}

#[test]
fn descriptor_validation_rejects_alignment_limits_and_incompatible_views() {
    let limits = wgpu::Limits::default();
    for spec in [
        BufferSpec {
            size: 0,
            usage: wgpu::BufferUsages::COPY_DST,
        },
        BufferSpec {
            size: 3,
            usage: wgpu::BufferUsages::COPY_DST,
        },
        BufferSpec {
            size: 4,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        },
        BufferSpec {
            size: 8,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE,
        },
        BufferSpec {
            size: 8,
            usage: wgpu::BufferUsages::from_bits_retain(u32::MAX),
        },
    ] {
        assert!(matches!(
            spec.bytes(&limits),
            Err(ResourceError::InvalidDescriptor(_))
        ));
    }
    let mut key = texture();
    key.view_format = Some(wgpu::TextureFormat::Rgba8Unorm);
    assert!(matches!(
        key.bytes(&limits),
        Err(ResourceError::InvalidDescriptor(_))
    ));
    key.format = wgpu::TextureFormat::Rgba8Unorm;
    key.view_format = Some(wgpu::TextureFormat::Rgba8UnormSrgb);
    assert_eq!(key.bytes(&limits).unwrap(), 128);
    key.size.width = limits.max_texture_dimension_2d + 1;
    assert!(key.bytes(&limits).is_err());
}

#[test]
fn texture_accounting_overflow_is_typed() {
    let limits = wgpu::Limits {
        max_texture_dimension_3d: u32::MAX,
        ..Default::default()
    };
    let mut key = texture();
    key.dimension = wgpu::TextureDimension::D3;
    key.size = wgpu::Extent3d {
        width: u32::MAX,
        height: u32::MAX,
        depth_or_array_layers: u32::MAX,
    };
    assert!(matches!(key.bytes(&limits), Err(ResourceError::Overflow)));
}

#[test]
fn full_descriptor_keys_distinguish_every_reuse_property() {
    let key = texture();
    let changes = [
        TextureSpec {
            size: wgpu::Extent3d {
                width: 4,
                ..key.size
            },
            ..key
        },
        TextureSpec {
            size: wgpu::Extent3d {
                height: 8,
                ..key.size
            },
            ..key
        },
        TextureSpec {
            size: wgpu::Extent3d {
                depth_or_array_layers: 2,
                ..key.size
            },
            ..key
        },
        TextureSpec {
            dimension: wgpu::TextureDimension::D3,
            ..key
        },
        TextureSpec {
            mip_level_count: 2,
            ..key
        },
        TextureSpec {
            sample_count: 4,
            ..key
        },
        TextureSpec {
            format: wgpu::TextureFormat::Rgba16Float,
            ..key
        },
        TextureSpec {
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            ..key
        },
        TextureSpec {
            view_format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..key
        },
    ];
    for changed in changes {
        assert_ne!(Spec::Texture(key), Spec::Texture(changed));
    }
    assert_eq!(
        TextureSpec {
            view_format: Some(key.format),
            ..key
        }
        .normalized(),
        key
    );
}

#[derive(Debug)]
struct Probe {
    bytes: u64,
    pool: Weak<ResourcePool<Probe>>,
    drops: Arc<AtomicUsize>,
}
impl Pooled for Probe {
    type Key = u64;
    fn key(&self) -> u64 {
        self.bytes
    }
    fn bytes(&self) -> u64 {
        self.bytes
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            let _ = pool.idle_count();
        }
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn pressure_eviction_preserves_policy_and_destroys_outside_lock() {
    let pool = Arc::new(ResourcePool::with_limits(100, 10));
    let drops = Arc::new(AtomicUsize::new(0));
    for bytes in [10, 20, 30] {
        pool.put(Probe {
            bytes,
            pool: Arc::downgrade(&pool),
            drops: drops.clone(),
        });
    }
    assert_eq!(pool.evict(11), 30);
    assert_eq!(pool.budget(), 100);
    assert_eq!(pool.idle_count(), (1, 30));
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    assert_eq!(pool.evict(0), 0);
}

fn context() -> WorkContext {
    let gpu = crate::shared_device().expect("GPU available");
    WorkContext::new(&gpu.device, &gpu.queue)
}
fn batch(context: &WorkContext, specs: &[Spec]) -> Batch {
    match context.resources().prepare(specs).unwrap().poll() {
        Poll::Ready(Ok(batch)) => batch,
        other => panic!("native scopes must settle: {other:?}"),
    }
}

#[test]
#[ignore = "requires GPU"]
fn gpu_batch_admission_reuse_rollback_and_context_identity() {
    let context = context();
    let resources = context.resources();
    let spec = Spec::Buffer(BufferSpec {
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST,
    });
    assert!(matches!(
        resources.prepare(&[spec]),
        Err(ResourceError::Admission(AdmissionError::Unconfigured))
    ));
    context.memory().set_limit(Some(128));
    resources.set_idle_budget(128);
    let mut ready = batch(&context, &[spec]);
    let buffer = ready.buffer(0).unwrap();
    assert!(ready.buffer(0).is_err());
    assert!(buffer.belongs_to(resources));
    assert!(buffer.belongs_to(context.clone().resources()));
    let other = WorkContext::new(context.device(), context.queue());
    assert!(!buffer.belongs_to(other.resources()));
    assert_eq!(buffer.bytes(), 64);
    assert_eq!(context.memory().snapshot().allocated, 64);
    let alias = buffer.clone();
    drop(buffer);
    assert_eq!(resources.idle(), (0, 0));
    drop(alias);
    assert_eq!(resources.idle(), (1, 64));
    context.memory().set_limit(None);
    let mut ready = batch(&context, &[spec]);
    assert_eq!(context.memory().snapshot().allocated, 64);
    drop(ready.buffer(0).unwrap());
    let large = Spec::Buffer(BufferSpec {
        size: 128,
        usage: wgpu::BufferUsages::COPY_DST,
    });
    assert!(matches!(
        resources.prepare(&[spec, large]),
        Err(ResourceError::Admission(AdmissionError::Unconfigured))
    ));
    assert_eq!(resources.idle(), (1, 64));
    context.memory().set_limit(Some(128));
    // The existing idle allocation is evicted under pressure without changing policy.
    let different = Spec::Buffer(BufferSpec {
        size: 128,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let mut ready = batch(&context, &[different]);
    assert_eq!(resources.0.pool.budget(), 128);
    assert_eq!(context.memory().snapshot().allocated, 128);
    let buffer = ready.buffer(0).unwrap();
    buffer.taint();
    drop(buffer);
    assert_eq!(resources.idle(), (0, 0));
    assert_eq!(context.memory().snapshot().allocated, 0);
    context.mark_lost("test terminal loss");
    assert!(matches!(
        resources.prepare(&[spec]),
        Err(ResourceError::Gpu(WaitError::DeviceLost(_)))
    ));
}

#[test]
#[ignore = "requires GPU"]
fn gpu_creation_failure_and_map_read_never_enter_idle_pool() {
    let context = context();
    context.memory().set_limit(Some(1 << 20));
    context.resources().set_idle_budget(1 << 20);
    // A descriptor with valid logical bytes but unsupported multisample usage
    // fails scoped native validation, retaining no partial new allocation.
    let mut invalid = texture();
    invalid.sample_count = 4;
    invalid.usage = wgpu::TextureUsages::COPY_SRC;
    let mut pending = context
        .resources()
        .prepare(&[Spec::Texture(invalid)])
        .unwrap();
    assert!(matches!(
        pending.poll(),
        Poll::Ready(Err(ResourceError::Gpu(_)))
    ));
    assert_eq!(context.memory().snapshot().used(), 0);
    let readback = Spec::Buffer(BufferSpec {
        size: 64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut ready = batch(&context, &[readback]);
    drop(ready.buffer(0).unwrap());
    assert_eq!(context.resources().idle(), (0, 0));
    assert_eq!(context.memory().snapshot().used(), 0);
}

struct Pins(Vec<Buffer>);
impl WorkLease for Pins {
    fn completed(self: Box<Self>, result: Result<(), WaitError>) {
        if result.is_err() {
            for buffer in &self.0 {
                buffer.taint();
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn gpu_staging_remap_is_retained_charged_and_reused_after_retirement() {
    let context = context();
    context.memory().set_limit(Some(128));
    context.resources().set_idle_budget(128);
    let staging_spec = Spec::Buffer(BufferSpec {
        size: 64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
    });
    let target_spec = Spec::Buffer(BufferSpec {
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    let mut ready = batch(&context, &[staging_spec, target_spec]);
    let mut staging = ready.buffer(0).unwrap();
    let target = ready.buffer(1).unwrap();
    let clone = staging.clone();
    assert!(matches!(
        staging.upload(0, &[7; 64]),
        Err(ResourceError::UploadState)
    ));
    drop(clone);
    staging.upload(0, &[7; 64]).unwrap();
    assert!(staging.upload(0, &[7; 64]).is_err());
    let work = RecordedWork::record(
        &context,
        || {
            let mut encoder = context.device().create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&staging, 0, &target, 0, 64);
            encoder.finish()
        },
        Pins(vec![staging.clone(), target.clone()]),
    )
    .unwrap();
    let mut queue = WorkQueue::new(&context, NonZeroUsize::new(1).unwrap());
    let _submission = queue.try_submit("staging test", work).unwrap();
    drop(staging);
    drop(target);
    assert_eq!(context.resources().idle(), (0, 0));
    queue.shutdown().unwrap();
    assert_eq!(context.memory().snapshot().allocated, 128);
    assert_eq!(context.resources().pending(), 1);
    crate::wait_idle(context.device(), context.queue()).unwrap();
    context.resources().collect().unwrap();
    assert_eq!(context.resources().pending(), 0);
    assert_eq!(context.resources().idle(), (2, 128));
    let before = context.memory().snapshot();
    let mut ready = batch(&context, &[staging_spec]);
    let mut staging = ready.buffer(0).unwrap();
    staging.upload(0, &[9; 64]).unwrap();
    assert_eq!(context.memory().snapshot().allocated, before.allocated);
    assert_eq!(context.memory().snapshot().peak, before.peak);
    // A true unsubmitted abort is safe to remap and reuse as well.
    drop(staging);
    assert_eq!(context.resources().pending(), 1);
}
