//! A high-performance data pipeline designed to pretokenize datasets and stream them directly to your training loop. It offers customizable data loading strategies and features a built-in zero-copy memory mapping strategy (`memmap2`) for maximum throughput.
//!
//! # Benchmark Scale
//!
//! * **Pretokenization Speed:** **~5.8M tokens/sec** (Processes FineWeb-Edu `sample-10B` in ~30 minutes).
//! * **DataLoader Streaming Speed:** Up to **10 GiB/s** (via sequential memory-mapped reads leveraging kernel page prefetching. Max throughput will further increase with memory pinning and async GPU host-to-device transfers to saturate PCIe bandwidth).
//!
//! > **Test Bench Setup:**
//! > * **CPU:** AMD Ryzen 7 5800X (8C / 16T)
//! > * **GPU:** NVIDIA GeForce RTX 3090 (24GB VRAM, PCIe 4.0 x16)
//! > * **RAM:** 32GB DDR4
//! > * **OS:** Linux (Kernel 7.0.10)
//!
//! # API Showcase
//!
//! ### 1. Streaming Data into an Execution Engine
//! Here is how you can use Plast to stream pretokenized data.
//!
//! ```rust, ignore
//! use plast::{MmapStorage, MmapSetup, AdviceSet, Dataloader, fetch_bin_files};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let context_window = 2048; // Example context window size
//!     let dataset_files = fetch_bin_files("pretokenized_dataset_dir")?;
//!     
//!     let setup = MmapSetup::new(dataset_files)
//!         .with_advice_set(AdviceSet::default());
//!     let storage = MmapStorage::load_data(setup)?;
//!     
//!     let mut loader = Dataloader::<_, BytesConverter>::new(storage, 1024);
//!
//!     // Option A: Get input and target slices shifted by 1 token
//!     for (i, (inputs, targets)) in loader.tf_iter(context_window).enumerate() {
//!         println!(
//!             "Ready for transfer to GPU! Batch size: {} tokens",
//!             inputs.len()
//!         );
//!         if i >= 10 {
//!             break;
//!         }
//!     }
//!     // Option B: you can iterate over the data sequentially
//!     for (i, inputs) in loader.iter(context_window).enumerate() {
//!         println!(
//!             "Ready for transfer to gpu and use in model! Batch size: {} tokens",
//!             inputs.len()
//!         );
//!         if i >= 10 {
//!             break;
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### 2. Pretokenizing a Dataset
//! Here is how you can use Plast to pretokenize raw data files.
//!
//! ```rust, ignore
//! use tokenizers::Tokenizer;
//! use plast::{ShardLoader, pretokenize_dataset, fetch_data_files};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//!     let data_files = fetch_data_files("data/")?;
//!     let shard_loader = ShardLoader::new(data_files, "text_column_name");
//!
//!     let eos_id = tokenizer
//!         .token_to_id("<|endoftext|>")
//!         .ok_or("No such token in tokenizer")?;
//!     
//!     let out_dir = "pretokenized_data/";
//!
//!     let _pretokenized_shards = pretokenize_dataset(
//!         &tokenizer,
//!         shard_loader,               
//!         out_dir,
//!         2 * 1024 * 1024 * 1024,     // Shard size in bytes (2 GiB)
//!         eos_id,                     // EOS Token ID
//!         2048,                       // Write queue capacity
//!     )?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! Plast is built on a clean 3-layer structure:
//! 1. **Storage**: Manages how data is stored, sliced, and loaded (e.g., zero-copy memory mapping via `memmap2` or buffered prefetching).
//! 2. **Dataloader**: An abstraction wrapper over Storage types that unifies and simplifies iterator creation.
//! 3. **Iterators**: Provides efficient, ready-to-use data streams (e.g., `tf_iter` for input/target pairs).
//!
//! # Feature Flags
//!
//! - `full`: Enables all features listed below except `legacy` and experimental features.
//! - `dataloader`: Enables `Dataloader` and iterators.
//! - `pretokenizer`: Enables `pretokenize_dataset`, `ShardLoader`, and related utilities.
//! - `mmap-storage`: Enables `MmapStorage` and initialization structs.
//! - `experimental-buffer-storage`: Enables `BufferStorage`.
//! - `burn`: Enables support for `Bytes` to `burn_tensor::Bytes` conversion.
//! - `legacy`: Enables the older API (`MmapPretokenizedDataLoader` and `MmapPretokenizedDataLoaderBytes`).

#[cfg(feature = "dataloader")]
pub mod dataloader;

#[cfg(feature = "legacy")]
pub mod legacy;

#[cfg(feature = "pretokenizer")]
pub mod pretokenizer;

pub mod storage;

// Always available core traits/structs
pub use storage::Storage;

// Dataloader feature re-exports
#[cfg(feature = "dataloader")]
pub use dataloader::{BytesConverter, Dataloader, DataloaderType};

#[cfg(all(feature = "dataloader", feature = "burn"))]
pub use dataloader::BurnBytesConverter;

#[cfg(feature = "legacy")]
pub use legacy::{MmapPretokenizedDataLoader, MmapPretokenizedDataLoaderBytes};

// Pretokenizer feature re-exports
#[cfg(feature = "pretokenizer")]
pub use pretokenizer::{
    ShardLoader, pretokenize_dataset,
    utils::{fetch_arrow_files, fetch_bin_files},
};

// Storage feature re-exports
#[cfg(feature = "experimental-buffer-storage")]
pub use storage::BufferStorage;

#[cfg(feature = "mmap-storage")]
pub use storage::{AdviceSet, MmapSetup, MmapStorage};
