#[cfg(feature = "dataloader")]
pub mod dataloader;
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

// Pretokenizer feature re-exports
#[cfg(feature = "pretokenizer")]
pub use pretokenizer::{ShardLoader, pretokenize_dataset, utils};

// Storage feature re-exports
#[cfg(feature = "experimental-buffer-storage")]
pub use storage::BufferStorage;

#[cfg(feature = "mmap-storage")]
pub use storage::{AdviceSet, MmapSetup, MmapStorage};
