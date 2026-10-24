//! The persistent pipeline cache of a device: compiled pipelines outlive the process.
//!
//! **Why:** a Vulkan driver compiles a large compute shader (the fractal body plus the Standard Surface BSDF) for
//! seconds on the CPU. Its own disk cache is keyed per application and is cold again in every new host (Natron,
//! Playa, Resolve load the same plug-in) and after it is evicted. A `wgpu::PipelineCache` whose data this crate keeps
//! on disk serves every process on the same adapter and driver: the first pipeline of a shader compiles once, every
//! later process loads it.
//!
//! **How:** [`pipeline_cache`](fn@crate::pipeline_cache::pipeline_cache) opens one cache per device (a process-wide registry), seeded from
//! `<cache dir>/gpu-info/wgpu-pipelines/<key>` with `key = wgpu::util::pipeline_cache_key` (adapter vendor, device
//! and backend); a pipeline is created with `cache: Some(cache.cache())` and [`DiskPipelineCache::save`] writes the
//! data back after a compile. `None` where there is nothing to cache: a device without
//! `wgpu::Features::PIPELINE_CACHE`, a backend without a cache key (only Vulkan has one in wgpu 30), no cache
//! directory. A missing, stale or foreign file is only a cache miss: the driver validates the header (vendor, device,
//! driver version, UUID) and wgpu falls back to an empty cache (`fallback: true`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A device's pipeline cache and the file it persists to.
pub struct DiskPipelineCache {
    cache: wgpu::PipelineCache,
    file: PathBuf,
    /// Content last written: rejected seeds can be larger than a fresh driver cache.
    saved: Mutex<Option<(usize, u64)>>,
}

impl std::fmt::Debug for DiskPipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskPipelineCache")
            .field("file", &self.file)
            .finish_non_exhaustive()
    }
}

impl DiskPipelineCache {
    /// The cache to pass as `cache` of a pipeline descriptor.
    pub fn cache(&self) -> &wgpu::PipelineCache {
        &self.cache
    }

    /// Write the cache's data when its content changed, including a smaller cache after a driver update:
    /// a sibling temporary file renamed over it, so a concurrent reader never sees half a file. A failure is logged
    /// and costs only the next process's compile.
    pub fn save(&self) {
        // Serialize snapshots with writes so an older concurrent snapshot cannot overwrite
        // a newer one. Size alone misses replacements after an incompatible seed is rejected.
        let mut saved = self
            .saved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(data) = self.cache.get_data() else {
            return;
        };
        let current = fingerprint(&data);
        if Some(current) == *saved {
            return;
        }
        match write_atomically(&self.file, &data) {
            Ok(()) => *saved = Some(current),
            Err(error) => log::warn!(
                "gpu-info: pipeline cache {} not written: {error}",
                self.file.display()
            ),
        }
    }
}

/// The pipeline cache of `device` (created from an adapter described by `adapter`), opened on first use and shared by
/// every later caller with the same device; `None` when the device cannot keep one (module docs).
pub fn pipeline_cache(
    device: &wgpu::Device,
    adapter: &wgpu::AdapterInfo,
) -> Option<Arc<DiskPipelineCache>> {
    static CACHES: Mutex<Vec<(wgpu::Device, Option<Arc<DiskPipelineCache>>)>> =
        Mutex::new(Vec::new());
    let mut caches = CACHES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, cache)) = caches.iter().find(|(known, _)| known == device) {
        return cache.clone();
    }
    let cache = open(device, adapter).map(Arc::new);
    caches.push((device.clone(), cache.clone()));
    cache
}

/// `device.create_compute_pipeline(desc)` through the device's persistent cache ([`pipeline_cache`], whatever `desc`
/// sets as `cache`), saving the cache after the compile: the one way the cluster's compute pipelines are created, so
/// each shader compiles once per adapter and driver rather than once per process.
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    adapter: &wgpu::AdapterInfo,
    desc: &wgpu::ComputePipelineDescriptor<'_>,
) -> wgpu::ComputePipeline {
    let cache = pipeline_cache(device, adapter);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        cache: cache.as_deref().map(DiskPipelineCache::cache),
        ..desc.clone()
    });
    if let Some(cache) = cache {
        cache.save();
    }
    pipeline
}

fn open(device: &wgpu::Device, adapter: &wgpu::AdapterInfo) -> Option<DiskPipelineCache> {
    open_in(device, adapter, &cache_dir()?)
}

/// [`open`] under `dir` instead of the user's cache directory.
fn open_in(
    device: &wgpu::Device,
    adapter: &wgpu::AdapterInfo,
    dir: &Path,
) -> Option<DiskPipelineCache> {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    let key = wgpu::util::pipeline_cache_key(adapter)?;
    let file = dir.join("gpu-info").join("wgpu-pipelines").join(key);
    let data = std::fs::read(&file).ok();
    // SAFETY: `data` is what `PipelineCache::get_data` of a device of this adapter key wrote (`save`), the only
    // contents this crate puts in that file; wgpu and the driver reject data of another wgpu version, adapter or
    // driver (`fallback: true` then starts empty), which is the documented contract of `create_pipeline_cache`.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("gpu-info disk pipeline cache"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    Some(DiskPipelineCache {
        cache,
        file,
        saved: Mutex::new(data.as_deref().map(fingerprint)),
    })
}

/// The user's cache directory: `%LOCALAPPDATA%` on Windows, `~/Library/Caches` on macOS, `$XDG_CACHE_HOME` or
/// `~/.cache` elsewhere.
fn cache_dir() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if cfg!(windows) {
        var("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| PathBuf::from(home).join("Library").join("Caches"))
    } else {
        var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| var("HOME").map(|home| PathBuf::from(home).join(".cache")))
    }
}

fn fingerprint(data: &[u8]) -> (usize, u64) {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut hash);
    (data.len(), hash.finish())
}

fn write_atomically(file: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let temp = file.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temp, data)?;
    std::fs::rename(&temp, file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_detects_equal_size_replacements_and_shrinkage() {
        assert_eq!(fingerprint(b"same"), fingerprint(b"same"));
        assert_ne!(fingerprint(b"old!"), fingerprint(b"new!"));
        assert_ne!(fingerprint(b"larger rejected seed"), fingerprint(b"new"));
    }

    /// wgpu rejects the outer header before passing bytes to the driver. A larger
    /// rejected seed must be replaced by the actual fresh cache, even without growth.
    #[test]
    #[ignore = "requires a Vulkan adapter with pipeline caching"]
    fn rejected_large_seed_is_replaced() {
        let gpu = crate::compute_device().expect("compute device");
        let info = gpu.adapter.get_info();
        let dir =
            std::env::temp_dir().join(format!("gpu-info-rejected-pcache-{}", std::process::id()));
        let first = open_in(&gpu.device, &info, &dir).expect("Vulkan cache");
        first.save();
        let mut rejected = std::fs::read(&first.file).expect("fresh cache persisted");
        rejected[0] ^= 0xff; // Invalid wgpu magic: never sent to Vulkan.
        rejected.resize(rejected.len() + 1024 * 1024, 0);
        std::fs::write(&first.file, &rejected).unwrap();
        let fresh = open_in(&gpu.device, &info, &dir).expect("fallback opens");
        let actual = fresh.cache().get_data().unwrap();
        assert!(actual.len() < rejected.len());
        fresh.save();
        assert!(
            std::fs::read(&fresh.file).unwrap() == actual,
            "Rejected seed was not replaced by the live cache"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A pipeline compiled through the cache is saved, and a cache reopened from the file starts with that data
    /// (what a later process gets); a second save of an unchanged cache writes nothing.
    #[test]
    #[ignore = "requires a Vulkan adapter with pipeline caching"]
    fn a_saved_cache_seeds_the_next_one() {
        let gpu = crate::compute_device().expect("compute device");
        let info = gpu.adapter.get_info();
        let dir = std::env::temp_dir().join(format!("gpu-info-pcache-{}", std::process::id()));
        let cache = open_in(&gpu.device, &info, &dir).expect("a Vulkan device keeps a cache");
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<storage, read_write> v: array<f32>;
                 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
                     v[i.x] = sqrt(v[i.x]) * 2.0 + 1.0;
                 }"
                    .into(),
                ),
            });
        let _pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Some(cache.cache()),
            });
        cache.save();
        let written = std::fs::read(&cache.file).expect("the cache file");
        assert!(!written.is_empty());
        let modified = std::fs::metadata(&cache.file).unwrap().modified().unwrap();
        cache.save();
        assert_eq!(
            std::fs::metadata(&cache.file).unwrap().modified().unwrap(),
            modified
        );
        let reopened = open_in(&gpu.device, &info, &dir).expect("reopen");
        assert_eq!(*reopened.saved.lock().unwrap(), Some(fingerprint(&written)));
        assert!(
            reopened
                .cache()
                .get_data()
                .is_some_and(|data| !data.is_empty())
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
