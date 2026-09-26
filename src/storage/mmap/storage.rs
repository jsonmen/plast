use super::super::error::DataLoaderError;
use super::setup::MmapSetup;
use crate::storage::Storage;
use bytes::Bytes;
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmapStorage {
    /// Vector of raw memory-mapped files.
    shards: Vec<Bytes>,
    /// Track logical capacity per shard measured in *elements* (4 bytes each).
    shard_lengths: Vec<usize>,
    /// Cumulative raw byte start offsets for calculating global locations.
    /// This vector always has a length of `shards.len() + 1`.
    shard_offsets: Vec<usize>,
    /// Total logical elements across all shards combined.
    total_size: usize,

    current_shard_idx: usize,
    local_cursor: usize,
}
impl MmapStorage {
    pub fn load_data(mmap_setup: MmapSetup) -> Result<Self, DataLoaderError> {
        let data_files = mmap_setup.data_files();
        let files_count = data_files.len();
        let mut shards = Vec::with_capacity(files_count);
        let mut shard_lengths = Vec::with_capacity(files_count);
        let mut shard_offsets = Vec::with_capacity(files_count + 1);

        let mut total_size = 0;
        let mut current_byte_offset = 0;

        for path in data_files {
            let f = mmap_setup.open_options().open(&path).map_err(|e| {
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
    pub fn load_data_mut(mmap_setup: MmapSetup) -> Result<Self, DataLoaderError> {
        let data_files = mmap_setup.data_files();
        let files_count = data_files.len();
        let mut shards = Vec::with_capacity(files_count);
        let mut shard_lengths = Vec::with_capacity(files_count);
        let mut shard_offsets = Vec::with_capacity(files_count + 1);

        let mut total_size = 0;
        let mut current_byte_offset = 0;

        for path in data_files {
            let f = mmap_setup.open_options().open(&path).map_err(|e| {
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
    pub fn shards(&self) -> impl Iterator<Item = &[u8]> {
        self.shards.iter().map(|shard| &shard[..])
    }

    #[inline]
    pub fn locate(&self, global_inx: usize) -> (usize, usize) {
        let idx = self.shard_offsets.partition_point(|&off| off <= global_inx) - 1;
        let local_offset = global_inx - self.shard_offsets[idx];
        (idx, local_offset)
    }

    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }
}
impl Storage for MmapStorage {
    fn len(&self) -> usize {
        self.total_size
    }
    fn clear_state(&mut self) -> () {
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
}
