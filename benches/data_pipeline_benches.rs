use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use plast::dataloader::{BYTES_PER_TOKEN, BytesConverter, Dataloader};
use plast::{BufferStorage, MmapSetup, MmapStorage, Storage, pretokenize_dataset};
use polars::prelude::*;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use tempfile::{NamedTempFile, TempDir};
use tokenizers::Tokenizer;

// CUDA specific driver bindings
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// Mock data generation helper
fn create_heavy_mock_dataset(rows: usize) -> StringChunked {
    let base_phrases = [
        "The quick brown fox jumps over the lazy dog near rust arrays.",
        "High performance data structures saturate storage bandwidth channels cleanly.",
        "Unsafe code blocks decouple execution architectures from safety monitors.",
    ];
    let values: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "{} Offset marker sequence counter: {}",
                base_phrases[i % 3],
                i
            )
        })
        .collect();

    Series::new("text".into(), values).str().unwrap().clone()
}

fn load_real_text_sample() -> StringChunked {
    // Read actual parquet file into string column
    // This sample.parquet is 013_00000.parquet from sample-10BT fineweb-edu
    let df = LazyFrame::scan_parquet(
        polars::prelude::PlRefPath::new("fixtures/sample.parquet"),
        Default::default(),
    )
    .unwrap()
    .slice(0, 40000)
    .collect()
    .unwrap();
    df.column("text").unwrap().str().unwrap().clone()
}

fn create_mock_tokenizer() -> Tokenizer {
    let large_json_data = include_str!("../fixtures/mock_tokenizer.json");

    let mut tmp_json_file = NamedTempFile::new().expect("Failed to create temp file");
    tmp_json_file
        .write_all(large_json_data.as_bytes())
        .expect("Failed to write mock data");
    tmp_json_file.flush().expect("Failed to flush file buffers");

    Tokenizer::from_file(tmp_json_file.path()).expect("Failed to load tokenizer")
}

/// 1. BENCHMARK: CPU Pretokenizer pipeline throughput
fn bench_pretokenizer(c: &mut Criterion) {
    let tokenizer = create_mock_tokenizer();
    let dataset = load_real_text_sample();

    // 1. Calculate raw input bytes
    let raw_bytes = dataset
        .iter()
        .flatten()
        .map(|s| s.len() as u64)
        .sum::<u64>();

    // 2. Compute exact token count produced by this dataset for exact Mtok/s metrics
    let total_tokens: u64 = dataset
        .iter()
        .flatten()
        .map(|s| tokenizer.encode(s, false).unwrap().get_ids().len() as u64)
        .sum();

    let dataset_ref = Arc::new(dataset);

    let mut group = c.benchmark_group("Pretokenizer_Performance");
    group.measurement_time(std::time::Duration::from_secs(15));
    group.sample_size(10);

    // --- Metric 1: Input Data Throughput (MiB/s or GiB/s) ---
    group.throughput(Throughput::Bytes(raw_bytes));
    group.bench_function("Input_Bytes_Throughput", |b| {
        b.iter_with_setup(
            || (TempDir::new().unwrap(), dataset_ref.clone()),
            |(tmp_dir, data)| {
                let iter = std::iter::once(Ok((*data).clone()));

                let _ = pretokenize_dataset(
                    &tokenizer,
                    iter,
                    tmp_dir.path(),
                    50 * 1024 * 1024,
                    50256,
                    2048,
                )
                .unwrap();
            },
        );
    });

    // --- Metric 2: Output Token Generation Rate (tok/s or Mtok/s) ---
    group.throughput(Throughput::Elements(total_tokens));
    group.bench_function("Output_Tokens_Throughput", |b| {
        b.iter_with_setup(
            || (TempDir::new().unwrap(), dataset_ref.clone()),
            |(tmp_dir, data)| {
                let iter = std::iter::once(Ok((*data).clone()));
                let _ = pretokenize_dataset(
                    &tokenizer,
                    iter,
                    tmp_dir.path(),
                    50 * 1024 * 1024,
                    50256,
                    2048,
                )
                .unwrap();
            },
        );
    });

    group.finish();
}

/// 2. BENCHMARK: H2D Bus Saturation & Kernel execution tracking
fn bench_gpu_saturation(c: &mut Criterion) {
    let tokenizer = create_mock_tokenizer();
    let dataset = create_heavy_mock_dataset(500_000);
    let tmp_dir = TempDir::new().unwrap();

    let paths = pretokenize_dataset(
        &tokenizer,
        vec![Ok(dataset)].into_iter(),
        tmp_dir.path(),
        100 * 1024 * 1024,
        50256,
        8,
    )
    .unwrap();

    let temp_storage = MmapStorage::load_data(MmapSetup::new(paths.clone())).unwrap();
    let total_elements = temp_storage.total_size();
    let total_bytes = total_elements * BYTES_PER_TOKEN;
    drop(temp_storage);

    let ctx = CudaContext::new(0).expect("Missing CUDA GPU device context execution capability.");
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_file("benches/kernels/sum.ptx"))
        .unwrap();
    let f = module.load_function("sum_tokens").unwrap();

    let context_window_elements = 4096 * 64;
    let context_window_bytes = context_window_elements * BYTES_PER_TOKEN;

    let gpu_vec = stream.alloc_zeros::<u32>(context_window_elements).unwrap();
    let mut dev_sum = stream.alloc_zeros::<u64>(1).unwrap();
    let iterations = total_elements / context_window_elements;

    let mut group = c.benchmark_group("GPU_Saturation");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // --- Strategy A: Random-Access Batching ---
    group.bench_function("H2D_Transfer_Plus_Reduction_SliceRandom", |b| {
        // 1. Create storage ONCE outside the timing loop
        let storage = MmapStorage::load_data(MmapSetup::new(paths.clone())).unwrap();

        b.iter_custom(|iters| {
            let start = Instant::now();

            // 2. Reuse the SAME storage instance across all iterations
            for _ in 0..iters {
                for step in 0..iterations {
                    let start_byte = step * context_window_bytes;
                    let end_byte = start_byte + context_window_bytes;

                    if let Some(input_bytes) = storage.slice_random(start_byte..end_byte) {
                        let raw_tokens: &[u32] = bytemuck::cast_slice(&input_bytes[..]);
                        let n = raw_tokens.len();

                        unsafe {
                            let (src, _record_src) = gpu_vec.device_ptr(&stream);
                            let _ = cudarc::driver::result::memcpy_htod_async(
                                src,
                                raw_tokens,
                                stream.cu_stream(),
                            );
                        };

                        let threads_per_block = 256;
                        let blocks_per_grid = n.div_ceil(threads_per_block) as u32;
                        let cfg = LaunchConfig {
                            grid_dim: (blocks_per_grid, 1, 1),
                            block_dim: (threads_per_block as u32, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launch_args = stream.launch_builder(&f);
                        launch_args.arg(&gpu_vec);
                        launch_args.arg(&mut dev_sum);
                        launch_args.arg(&n);

                        unsafe { launch_args.launch(cfg) }.unwrap();
                    }
                }
            }
            ctx.synchronize().unwrap();
            start.elapsed()
        });
    });

    // --- Strategy B: Sequential Zero-Copy Streaming ---
    group.bench_function("H2D_Transfer_Plus_Reduction_DataloaderIter", |b| {
        let storage =
            MmapStorage::load_data_mut(MmapSetup::new(paths.clone()).read_write()).unwrap();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let iter_storage = storage.clone();
                let mut dataloader = Dataloader::<MmapStorage, BytesConverter>::new(
                    iter_storage,
                    context_window_elements,
                );

                for input_bytes in dataloader.iter_bytes() {
                    let raw_tokens: &[u32] = bytemuck::cast_slice(&input_bytes[..]);
                    let n = raw_tokens.len();

                    unsafe {
                        let (src, _record_src) = gpu_vec.device_ptr(&stream);
                        cudarc::driver::result::memcpy_htod_async(
                            src,
                            raw_tokens,
                            stream.cu_stream(),
                        )
                        .unwrap();
                    };

                    let threads_per_block = 256;
                    let blocks_per_grid = n.div_ceil(threads_per_block) as u32;
                    let cfg = LaunchConfig {
                        grid_dim: (blocks_per_grid, 1, 1),
                        block_dim: (threads_per_block as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let num = n as i32;
                    let mut launch_args = stream.launch_builder(&f);
                    launch_args.arg(&gpu_vec);
                    launch_args.arg(&mut dev_sum);
                    launch_args.arg(&num);

                    unsafe { launch_args.launch(cfg) }.unwrap();
                }
            }

            ctx.synchronize().unwrap();
            start.elapsed()
        });

        // --- UNREGISTER ALL SHARDS ---
        for shard in storage.shards() {
            unsafe {
                let ptr = shard.as_ptr() as *mut std::ffi::c_void;
                cudarc::driver::sys::cuMemHostUnregister(ptr).result().ok();
            }
        }
    });

    group.finish();
}

/// 3. BENCHMARK: BufferStorage H2D Bus Saturation & Kernel execution tracking
fn bench_buffer_storage_gpu_saturation(c: &mut Criterion) {
    let tokenizer = create_mock_tokenizer();
    let dataset = create_heavy_mock_dataset(500_000);
    let tmp_dir = TempDir::new().unwrap();

    let paths = pretokenize_dataset(
        &tokenizer,
        vec![Ok(dataset)].into_iter(),
        tmp_dir.path(),
        100 * 1024 * 1024, // 100 MB shard size
        50256,
        8,
    )
    .unwrap();

    // Calculate total size from file metadata since BufferStorage doesn't expose it before loading
    let total_bytes: u64 = paths
        .iter()
        .map(|p| std::fs::metadata(p).unwrap().len())
        .sum();
    let total_elements = (total_bytes / BYTES_PER_TOKEN as u64) as usize;

    let ctx = CudaContext::new(0).expect("Missing CUDA GPU device context execution capability.");
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_file("benches/kernels/sum.ptx"))
        .unwrap();
    let f = module.load_function("sum_tokens").unwrap();

    let context_window_elements = 4096 * 64;
    let context_window_bytes = context_window_elements * BYTES_PER_TOKEN;

    let gpu_vec = stream.alloc_zeros::<u32>(context_window_elements).unwrap();
    let mut dev_sum = stream.alloc_zeros::<u64>(1).unwrap();
    let iterations = total_elements / context_window_elements;

    let mut group = c.benchmark_group("BufferStorage_GPU_Saturation");
    group.throughput(Throughput::Bytes(total_bytes));

    // --- Strategy A: Manual Sequential Batching ---
    // This strategy uses `slice_sequential` in a manual loop to simulate batched fetching.
    group.bench_function("H2D_Transfer_Plus_Reduction_ManualSequential", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();

            for _ in 0..iters {
                // IMPORTANT: Unlike MmapStorage, BufferStorage is a single-pass consumer
                // (it drains the crossbeam channel). We MUST recreate it per iteration
                // to measure the full pipeline cost (file open + mmap + thread spawn + read).
                let mut storage = BufferStorage::load_data(paths.clone(), 8).unwrap();
                let mut steps_done = 0;

                while steps_done < iterations {
                    if let Some(input_bytes) = storage.slice_sequential(context_window_bytes) {
                        let raw_tokens: &[u32] = bytemuck::cast_slice(&input_bytes[..]);
                        let n = raw_tokens.len();

                        unsafe {
                            let (src, _record_src) = gpu_vec.device_ptr(&stream);
                            let _ = cudarc::driver::result::memcpy_htod_async(
                                src,
                                raw_tokens,
                                stream.cu_stream(),
                            );
                        };

                        let threads_per_block = 256;
                        let blocks_per_grid = n.div_ceil(threads_per_block) as u32;
                        let cfg = LaunchConfig {
                            grid_dim: (blocks_per_grid, 1, 1),
                            block_dim: (threads_per_block as u32, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launch_args = stream.launch_builder(&f);
                        launch_args.arg(&gpu_vec);
                        launch_args.arg(&mut dev_sum);
                        launch_args.arg(&n);

                        unsafe { launch_args.launch(cfg) }.unwrap();
                        steps_done += 1;
                    } else {
                        // End of data reached
                        break;
                    }
                }
            }
            ctx.synchronize().unwrap();
            start.elapsed()
        });
    });

    // --- Strategy B: Sequential Zero-Copy Streaming via Dataloader Iterator ---
    group.bench_function("H2D_Transfer_Plus_Reduction_DataloaderIter", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();

            for _ in 0..iters {
                // IMPORTANT: Recreate storage per iteration because BufferStorage is single-pass.
                // Buffer size of 4 allows the background thread to prefetch up to 4 shards.
                let storage = BufferStorage::load_data(paths.clone(), 8).unwrap();

                // Dataloader consumes the storage, satisfying the API design
                let mut dataloader = Dataloader::<BufferStorage, BytesConverter>::new(
                    storage,
                    context_window_elements,
                );

                for input_bytes in dataloader.iter_bytes() {
                    let raw_tokens: &[u32] = bytemuck::cast_slice(&input_bytes[..]);
                    let n = raw_tokens.len();

                    unsafe {
                        let (src, _record_src) = gpu_vec.device_ptr(&stream);
                        let _ = cudarc::driver::result::memcpy_htod_async(
                            src,
                            raw_tokens,
                            stream.cu_stream(),
                        );
                    };

                    let threads_per_block = 256;
                    let blocks_per_grid = n.div_ceil(threads_per_block) as u32;
                    let cfg = LaunchConfig {
                        grid_dim: (blocks_per_grid, 1, 1),
                        block_dim: (threads_per_block as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launch_args = stream.launch_builder(&f);
                    launch_args.arg(&gpu_vec);
                    launch_args.arg(&mut dev_sum);
                    launch_args.arg(&n);

                    unsafe { launch_args.launch(cfg) }.unwrap();
                }
            }
            ctx.synchronize().unwrap();

            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pretokenizer,
    bench_gpu_saturation,
    bench_buffer_storage_gpu_saturation
);
criterion_main!(benches);
