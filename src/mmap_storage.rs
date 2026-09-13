use crate::errors::DataLoaderError;
use crate::storage::Storage;
use bytes::Bytes;
use memmap2::Mmap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
#[derive(Clone)]
struct MmapOwner(Arc<Mmap>);

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}
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
}
impl MmapStorage {
    pub fn load_data<P: AsRef<Path>>(data_files: Vec<P>) -> Result<Self, DataLoaderError> {
        let files_count = data_files.len();
        let mut shards = Vec::with_capacity(files_count);
        let mut shard_lengths = Vec::with_capacity(files_count);
        let mut shard_offsets = Vec::with_capacity(files_count + 1);

        let mut total_size = 0;
        let mut current_byte_offset = 0;

        for path_ref in data_files {
            let path = path_ref.as_ref();
            let f = File::open(path).map_err(|e| DataLoaderError::ShardOpenFailed {
                source: e,
                path: path.to_path_buf(),
            })?;

            // SAFETY: Memory mapping is inherently unsafe because the underlying file
            // can be modified externally, causing undefined behavior in the process.
            let mmap =
                unsafe { Mmap::map(&f) }.map_err(|e| DataLoaderError::MemoryMappingFailed {
                    source: e,
                    path: path.to_path_buf(),
                })?;

            // Optimize for sequential access over large files via OS huge pages
            let _ = mmap.advise(memmap2::Advice::HugePage);

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
            let owner = MmapOwner(Arc::new(mmap));

            let bytes = Bytes::from_owner(owner);
            shards.push(bytes);
        }

        // Push final terminal boundary for the binary search interval math
        shard_offsets.push(current_byte_offset);

        Ok(Self {
            shards,
            shard_lengths,
            shard_offsets,
            total_size,
        })
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

    fn slice_sequential(
        &self,
        req_len: usize,
        current_shard_idx: &mut usize,
        local_cursor: &mut usize,
    ) -> Option<Bytes> {
        while *current_shard_idx < self.shards.len() {
            let active_shard = &self.shards[*current_shard_idx];

            if *local_cursor + req_len <= active_shard.len() {
                let start = *local_cursor;
                *local_cursor += req_len;
                return Some(active_shard.slice(start..start + req_len));
            }

            *current_shard_idx += 1;
            *local_cursor = 0;
        }

        None
    }
}
