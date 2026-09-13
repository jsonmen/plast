use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use plast::dataloader::{BYTES_PER_TOKEN, Dataloader};
use plast::datatypes::BytesConverter;
use plast::mmap_storage::MmapStorage;
use plast::pretokenize_dataset;
use plast::storage::Storage;
use polars::prelude::*;
use std::io::Write;
use std::time::Instant;
use tempfile::{NamedTempFile, TempDir};
use tokenizers::Tokenizer;

// CUDA specific driver bindings
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// Mock data generation helper
fn create_heavy_mock_dataset(rows: usize) -> StringChunked {
    let base_phrases = vec![
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
    let dataset = create_heavy_mock_dataset(200_000);

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

    let mut group = c.benchmark_group("Pretokenizer_Performance");
    group.measurement_time(std::time::Duration::from_secs(15));
    group.sample_size(20);

    // --- Metric 1: Input Data Throughput (MiB/s or GiB/s) ---
    group.throughput(Throughput::Bytes(raw_bytes));
    group.bench_function("Input_Bytes_Throughput", |b| {
        b.iter_with_setup(
            || TempDir::new().unwrap(),
            |tmp_dir| {
                let _ = pretokenize_dataset(
                    &tokenizer,
                    vec![Ok(dataset.clone())].into_iter(),
                    tmp_dir.path(),
                    50 * 1024 * 1024,
                    50256,
                    8,
                )
                .unwrap();
            },
        );
    });

    // --- Metric 2: Output Token Generation Rate (tok/s or Mtok/s) ---
    group.throughput(Throughput::Elements(total_tokens));
    group.bench_function("Output_Tokens_Throughput", |b| {
        b.iter_with_setup(
            || TempDir::new().unwrap(),
            |tmp_dir| {
                let _ = pretokenize_dataset(
                    &tokenizer,
                    vec![Ok(dataset.clone())].into_iter(),
                    tmp_dir.path(),
                    50 * 1024 * 1024,
                    50256,
                    8,
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

    // Pretokenize data shards
    let paths = pretokenize_dataset(
        &tokenizer,
        vec![Ok(dataset)].into_iter(),
        tmp_dir.path(),
        100 * 1024 * 1024,
        50256,
        8,
    )
    .unwrap();

    // Context size calculations
    let temp_storage = MmapStorage::load_data(paths.clone()).unwrap();
    let total_elements = temp_storage.total_size();
    let total_bytes = total_elements * BYTES_PER_TOKEN;
    drop(temp_storage);

    // Initialize CUDA hardware context handles
    let ctx = CudaContext::new(0).expect("Missing CUDA GPU device context execution capability.");
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_file("benches/kernels/sum.ptx"))
        .unwrap();
    let f = module.load_function("sum_tokens").unwrap();

    let context_window_elements = 4096;
    let context_window_bytes = context_window_elements * BYTES_PER_TOKEN;

    let gpu_vec = stream.alloc_zeros::<u32>(context_window_elements).unwrap();
    let mut dev_sum = stream.alloc_zeros::<u64>(1).unwrap();
    let iterations = total_elements / context_window_elements;

    let mut group = c.benchmark_group("GPU_Saturation");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // --- Strategy A: Random-Access Batching via `storage.slice_random` ---
    group.bench_function("H2D_Transfer_Plus_Reduction_SliceRandom", |b| {
        b.iter_custom(|iters| {
            // Pre-create `iters` storage instances so creation time isn't measured
            let storages: Vec<_> = (0..iters)
                .map(|_| MmapStorage::load_data(paths.clone()).unwrap())
                .collect();

            let start = Instant::now();

            for storage in storages {
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
                        let blocks_per_grid =
                            ((n + threads_per_block - 1) / threads_per_block) as u32;
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

    // --- Strategy B: Sequential Zero-Copy Streaming via `Dataloader` Iterator ---
    group.bench_function("H2D_Transfer_Plus_Reduction_DataloaderIter", |b| {
        b.iter_custom(|iters| {
            // Pre-allocate 'iters' owned storage objects before starting the clock
            let storages: Vec<_> = (0..iters)
                .map(|_| MmapStorage::load_data(paths.clone()).unwrap())
                .collect();

            let start = Instant::now();

            for storage in storages {
                // Owned `storage` is passed by value and consumed here
                let dataloader = Dataloader::<MmapStorage, BytesConverter>::new(
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
                    let blocks_per_grid = ((n + threads_per_block - 1) / threads_per_block) as u32;
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

criterion_group!(benches, bench_pretokenizer, bench_gpu_saturation);
criterion_main!(benches);
