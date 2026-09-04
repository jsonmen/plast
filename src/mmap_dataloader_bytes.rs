use crate::errors::DataLoaderError;
use bytes::Bytes;
use memmap2::Mmap;
use std::fs::File;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
struct MmapOwner(Arc<Mmap>);

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0[..]
    }
}

/// A high-performance data loader that memory-maps multiple pretokenized shards
/// of data, exposing contiguous slice views across underlying bytes.
///
/// It assumes every element in the dataset is a 4-byte token (e.g., `u32` or `i32`).
#[derive(Debug)]
pub struct MmapPretokenizedDataLoaderBytes {
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

impl MmapPretokenizedDataLoaderBytes {
    /// Maps a collection of data files into memory.
    ///
    /// # Arguments
    /// * `data_files` - A collection of paths to files containing 4-byte aligned data.
    ///
    /// # Errors
    /// Returns `DataLoaderError` if a file cannot be opened, memory mapping fails,
    /// or if a file size is not a multiple of 4 bytes.
    pub fn map_data<P: AsRef<Path>>(data_files: Vec<P>) -> Result<Self, DataLoaderError> {
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

    /// Returns the total number of logical elements (4-byte chunks) across all shards.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Returns the total number of open memory-mapped shards.
    #[inline]
    pub fn total_num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Access structural element counts per shard.
    #[inline]
    pub fn shard_lengths(&self) -> &[usize] {
        &self.shard_lengths
    }

    /// Access cumulative byte offsets mapping out shard boundaries.
    #[inline]
    pub fn shard_offsets(&self) -> &[usize] {
        &self.shard_offsets
    }

    pub fn iter(&self, num_elements: usize) -> ChunkedShardBytesIter<'_, &'_ [u8]> {
        ChunkedShardBytesIter::new(self, num_elements)
    }
    pub fn iter_tf(&self, num_elements: usize) -> ChunkedShardBytesIter<'_, (Bytes, Bytes)> {
        ChunkedShardBytesIter::new(self, num_elements)
    }
}

/// A generic zero-copy iterator over `PretokenizedDataLoader` shards.
pub struct ChunkedShardBytesIter<'a, T> {
    dataloader: &'a MmapPretokenizedDataLoaderBytes,
    current_idx: usize, // Element tracking relative to the *current* active shard
    current_file_idx: usize,
    num_elements: usize,
    _marker: PhantomData<T>,
}

impl<'a, T> ChunkedShardBytesIter<'a, T> {
    fn new(dataloader: &'a MmapPretokenizedDataLoaderBytes, num_elements: usize) -> Self {
        Self {
            dataloader,
            current_idx: 0,
            current_file_idx: 0,
            num_elements,
            _marker: PhantomData,
        }
    }
}

impl<'a> Iterator for ChunkedShardBytesIter<'a, &'a [u8]> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // Loop tracks file boundaries, rolling over to the next shard if the current one is exhausted
        while self.current_file_idx < self.dataloader.total_num_shards() {
            let active_shard = &self.dataloader.shards[self.current_file_idx];
            let shard_len_elements = self.dataloader.shard_lengths[self.current_file_idx];

            // CONFUSING MOMENT: Evaluating remainder windows.
            // If the current window fits entirely within the remaining space of this shard, return it.
            if self.current_idx + self.num_elements <= shard_len_elements {
                let start_byte = self.current_idx * 4;
                let end_byte = start_byte + (self.num_elements * 4);
                let byte_window = &active_shard[start_byte..end_byte];

                self.current_idx += self.num_elements;

                // Zero-copy transformation using `bytemuck` across types bounded by POD
                return Some(bytemuck::cast_slice(byte_window));
            }

            // If the remaining window straddles a shard boundary, we drop the remainder chunk
            // and advance to clean alignments on the next shard.
            self.current_file_idx += 1;
            self.current_idx = 0;
        }

        None
    }
}
impl<'a> Iterator for ChunkedShardBytesIter<'a, (Bytes, Bytes)> {
    type Item = (Bytes, Bytes);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_file_idx < self.dataloader.total_num_shards() {
            let active_shard = &self.dataloader.shards[self.current_file_idx];
            let shard_byte_len = self.dataloader.shard_lengths[self.current_file_idx] * 4;

            let local_start = self.current_idx * 4;
            let batch_byte_len = self.num_elements * 4;

            // Check if input + 4-byte target shift fits inside the current shard boundary
            if local_start + batch_byte_len + 4 <= shard_byte_len {
                let input = active_shard.slice(local_start..local_start + batch_byte_len);
                let target = active_shard.slice(local_start + 4..local_start + batch_byte_len + 4);

                self.current_idx += self.num_elements;
                return Some((input, target));
            }

            // Move to the next shard if the request straddles the boundary or overflows
            self.current_file_idx += 1;
            self.current_idx = 0;
        }

        None
    }
}
