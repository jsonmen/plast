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

#[derive(Error, Debug)]
pub enum PretokenizerError {
    #[error("Failed to create target directory structure at path: {path}")]
    DirectoryCreationFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Failed to create shard file at path: {path}")]
    ShardCreationFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Failed to write token payload data to shard file at path: {path}")]
    WriteFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Failed to flush buffered writer for shard file at path: {path}")]
    FlushFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },
}

#[derive(Error, Debug)]
pub enum ShardLoaderError {
    #[error("Failed to open dataset file at path: {path}")]
    FileOpenFailed {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Failed to parse file via Polars engine: {path}")]
    PolarsParseFailed {
        #[source]
        source: polars::prelude::PolarsError,
        path: PathBuf,
    },

    #[error("Required column '{column}' was not found in dataset file schema: {path}")]
    ColumnMissing { column: String, path: PathBuf },

    #[error("Column '{column}' exists but is not of expected String type in file: {path}")]
    InvalidColumnType { column: String, path: PathBuf },

    #[error("Unsupported or missing file extension for file: {path}")]
    UnsupportedExtension { path: PathBuf },

    #[error("Invalid path representation for file: {path}")]
    InvalidPath { path: PathBuf },
}
