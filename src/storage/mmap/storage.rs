//! Memory-mapped storage backend for high-performance data loading.

use super::super::error::DataLoaderError;
use super::setup::MmapSetup;
use crate::storage::Storage;
use bytes::Bytes;
use std::ops::Range;

/// A storage backend that utilizes memory-mapped files (`mmap`) for high-throughput data access.
///
/// `MmapStorage` maps one or more data files directly into the process's virtual address space.
/// This avoids expensive kernel-to-user space data copies and allows the operating system to
/// manage paging and caching efficiently. It supports both sequential iteration and random access slicing.
///
/// # Examples
///
/// ```rust,ignore
/// use plast::storage::{MmapStorage, setup::MmapSetup};
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let setup = MmapSetup::new(vec!["data/shard1.bin", "data/shard2.bin"]);
/// let storage = MmapStorage::load_data(setup)?;
///
/// println!("Total elements loaded: {}", storage.total_size());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmapStorage {
    /// Vector of raw memory-mapped files.
    shards: Vec<Bytes>,
    /// Track capacity per shard measured in *elements* (4 bytes each).
    shard_lengths: Vec<usize>,
    /// Cumulative raw byte start offsets for calculating global locations.
    /// This vector always has a length of `shards.len() + 1`.
    shard_offsets: Vec<usize>,
    /// Total elements across all shards combined.
    total_size: usize,

    current_shard_idx: usize,
    local_cursor: usize,
}

impl MmapStorage {
    /// Loads the specified files into memory-mapped storage in read-only mode.
    ///
    /// This method applies the memory advice specified in the `MmapSetup` (defaulting to
    /// optimized Linux sequential/hugepage settings) and validates that the files are
    /// perfectly aligned to 4-byte boundaries.
    ///
    /// # Errors
    /// Returns a `DataLoaderError` if a file cannot be opened, memory mapping fails,
    /// advice application fails, or if a file's size is not a multiple of 4 bytes.
    pub fn load_data(mmap_setup: MmapSetup) -> Result<Self, DataLoaderError> {
        let data_files = mmap_setup.data_files();
        let files_count = data_files.len();
        let mut shards = Vec::with_capacity(files_count);
        let mut shard_lengths = Vec::with_capacity(files_count);
        let mut shard_offsets = Vec::with_capacity(files_count + 1);

        let mut total_size = 0;
        let mut current_byte_offset = 0;

        for path in data_files {
            let f = mmap_setup.open_options().open(path).map_err(|e| {
                DataLoaderError::ShardOpenFailed {
                    source: e,
                    path: path.to_path_buf(),
                }
            })?;

            // SAFETY: Memory mapping is inherently unsafe because the underlying file
            // can be modified externally, causing undefined behavior in the process.
            let mmap = unsafe { mmap_setup.mmap_options().map(&f) }.map_err(|e| {
                DataLoaderError::MemoryMappingFailed {
                    source: e,
                    path: path.to_path_buf(),
                }
            })?;

            mmap_setup.advice_set().apply(&mmap)?;
            let byte_len = mmap.len();

            // CRITICAL: Ensure binary layout matches 4-byte boundaries (u32/i32)
            if byte_len % 4 != 0 {
                return Err(DataLoaderError::InvalidByteAlignment {
                    size: byte_len,
                    path: path.to_path_buf(),
                });
            }

            let num_elements = byte_len / 4;

            shard_offsets.push(current_byte_offset);
            current_byte_offset += byte_len;

            total_size += num_elements;
            shard_lengths.push(num_elements);

            let bytes = Bytes::from_owner(mmap);
            shards.push(bytes);
        }

        // Push final terminal boundary for the binary search interval math
        shard_offsets.push(current_byte_offset);

        Ok(Self {
            shards,
            shard_lengths,
            shard_offsets,
            total_size,
            current_shard_idx: 0,
            local_cursor: 0,
        })
    }

    /// Loads the specified files into memory-mapped storage in read-write mode.
    ///
    /// Similar to `load_data`, but allows modifications to the memory map to be
    /// persisted to the underlying files.
    pub fn load_data_mut(mmap_setup: MmapSetup) -> Result<Self, DataLoaderError> {
        let data_files = mmap_setup.data_files();
        let files_count = data_files.len();
        let mut shards = Vec::with_capacity(files_count);
        let mut shard_lengths = Vec::with_capacity(files_count);
        let mut shard_offsets = Vec::with_capacity(files_count + 1);

        let mut total_size = 0;
        let mut current_byte_offset = 0;

        for path in data_files {
            let f = mmap_setup.open_options().open(path).map_err(|e| {
                DataLoaderError::ShardOpenFailed {
                    source: e,
                    path: path.to_path_buf(),
                }
            })?;

            // SAFETY: Memory mapping is inherently unsafe because the underlying file
            // can be modified externally, causing undefined behavior in the process.
            let mmap = unsafe { mmap_setup.mmap_options().map_mut(&f) }.map_err(|e| {
                DataLoaderError::MemoryMappingFailed {
                    source: e,
                    path: path.to_path_buf(),
                }
            })?;

            mmap_setup.advice_set().apply_mut(&mmap)?;
            let byte_len = mmap.len();

            // CRITICAL: Ensure binary layout matches 4-byte boundaries (u32/i32)
            if byte_len % 4 != 0 {
                return Err(DataLoaderError::InvalidByteAlignment {
                    size: byte_len,
                    path: path.to_path_buf(),
                });
            }

            let num_elements = byte_len / 4;

            shard_offsets.push(current_byte_offset);
            current_byte_offset += byte_len;

            total_size += num_elements;
            shard_lengths.push(num_elements);

            let bytes = Bytes::from_owner(mmap);
            shards.push(bytes);
        }

        // Push final terminal boundary for the binary search interval math
        shard_offsets.push(current_byte_offset);

        Ok(Self {
            shards,
            shard_lengths,
            shard_offsets,
            total_size,
            current_shard_idx: 0,
            local_cursor: 0,
        })
    }

    /// Returns an iterator over the raw byte slices of each loaded shard.
    pub fn shards(&self) -> impl Iterator<Item = &[u8]> {
        self.shards.iter().map(|shard| &shard[..])
    }

    /// Locates which shard contains the given global byte index and calculates the local offset.
    ///
    /// # Arguments
    /// * `global_inx` - The global byte index across all concatenated shards.
    ///
    /// # Returns
    /// A tuple `(shard_index, local_byte_offset)`.
    #[inline]
    pub fn locate(&self, global_inx: usize) -> (usize, usize) {
        let idx = self.shard_offsets.partition_point(|&off| off <= global_inx) - 1;
        let local_offset = global_inx - self.shard_offsets[idx];
        (idx, local_offset)
    }

    /// Returns the total number of elements across all shards combined.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }
}

impl Storage for MmapStorage {
    fn len(&self) -> usize {
        self.total_size
    }

    fn clear_state(&mut self) {
        self.local_cursor = 0;
        self.current_shard_idx = 0;
    }

    fn slice_random(&self, range: Range<usize>) -> Option<Bytes> {
        let req_len = range.end - range.start;
        let total_bytes = *self.shard_offsets.last().unwrap_or(&0);

        if req_len == 0 || range.end > total_bytes {
            return None;
        }

        let (shard_idx, local_offset) = self.locate(range.start);
        let active_shard = &self.shards[shard_idx];

        if local_offset + req_len <= active_shard.len() {
            Some(active_shard.slice(local_offset..local_offset + req_len))
        } else {
            None
        }
    }

    fn slice_sequential(&mut self, req_len: usize) -> Option<Bytes> {
        while self.current_shard_idx < self.shards.len() {
            let active_shard = &self.shards[self.current_shard_idx];

            if self.local_cursor + req_len <= active_shard.len() {
                let start = self.local_cursor;
                self.local_cursor += req_len;
                return Some(active_shard.slice(start..start + req_len));
            }

            self.current_shard_idx += 1;
            self.local_cursor = 0;
        }

        None
    }
    fn slice_tf_sequential(&mut self, req_len: usize) -> Option<(Bytes, Bytes)> {
        while self.current_shard_idx < self.shards.len() {
            let active_shard = &self.shards[self.current_shard_idx];

            if self.local_cursor + req_len + 4 <= active_shard.len() {
                let start = self.local_cursor;
                self.local_cursor += req_len;
                return Some((
                    active_shard.slice(start..start + req_len),
                    active_shard.slice(start + 4..start + req_len + 4),
                ));
            }

            self.current_shard_idx += 1;
            self.local_cursor = 0;
        }

        None
    }
    fn slice_tf_random(&self, range: Range<usize>) -> Option<(Bytes, Bytes)> {
        let req_len = range.end - range.start;
        let total_bytes = *self.shard_offsets.last().unwrap_or(&0);

        if req_len == 0 || range.end > total_bytes {
            return None;
        }

        let (shard_idx, local_offset) = self.locate(range.start);
        let active_shard = &self.shards[shard_idx];

        if local_offset + req_len + 4 <= active_shard.len() {
            Some((
                active_shard.slice(local_offset..local_offset + req_len),
                active_shard.slice(local_offset + 4..local_offset + req_len + 4),
            ))
        } else {
            None
        }
    }
}
