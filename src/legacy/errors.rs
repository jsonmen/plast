use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Failed to open data file for memory mapping at path: {path}")]
    ShardOpenFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Failed to initialize virtual memory map allocation (mmap) for file: {path}")]
    MemoryMappingFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error(
        "File size ({size} bytes) is misaligned; must be a perfect multiple of 4 bytes. File: {path}"
    )]
    InvalidByteAlignment { size: usize, path: PathBuf },
}
