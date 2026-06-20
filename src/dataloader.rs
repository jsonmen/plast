use crate::errors::DataLoaderError;
use bytemuck::Pod;
use memmap2::Mmap;
use std::fs::File;
use std::marker::PhantomData;
use std::path::Path;

/// A high-performance data loader that memory-maps multiple pretokenized shards
/// of data, exposing contiguous slice views across underlying bytes.
///
/// It assumes every element in the dataset is a 4-byte token (e.g., `u32` or `i32`).
pub struct PretokenizedDataLoader {
    /// Vector of raw memory-mapped files.
    shards: Vec<Mmap>,
    /// Track logical capacity per shard measured in *elements* (4 bytes each).
    shard_lengths: Vec<usize>,
    /// Cumulative raw byte start offsets for calculating global locations.
    /// This vector always has a length of `shards.len() + 1`.
    shard_offsets: Vec<usize>,
    /// Total logical elements across all shards combined.
    total_size: usize,
}

impl PretokenizedDataLoader {
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
            shards.push(mmap);
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

    /// Retrieves input and target byte slices using u32-element based indexing.
    ///
    /// This splits a contiguous window into current context (`input`) and shifted targets (`target`).
    ///
    /// # Arguments
    /// * `idx` - Logical sequence/batch multiplier.
    /// * `batch_elements` - Sequence context window size in element count.
    ///
    /// # Examples
    /// ```no_run
    /// # use plast::PretokenizedDataLoader;
    /// # let loader = PretokenizedDataLoader::map_data(vec!["shard.bin"]).unwrap();
    /// if let Some((input, target)) = loader.get_tf_batch_u8(0, 1024) {
    ///     assert_eq!(input.len(), 4096);
    ///     assert_eq!(target.len(), 4096);
    /// }
    /// ```
    pub fn get_tf_batch_u8(&self, idx: usize, batch_elements: usize) -> Option<(&[u8], &[u8])> {
        let batch_byte_len = batch_elements * 4;
        let global_byte_offset = idx * batch_byte_len;

        // CONFUSING MOMENT: Finding which shard contains our global offset.
        // `binary_search` searches for the absolute byte position.
        // If an exact match is hit (Ok), that's our shard index.
        // If it falls between boundaries (Err(idx)), the correct shard is the one right before it (idx - 1).
        let shard_idx = match self.shard_offsets.binary_search(&global_byte_offset) {
            Ok(exact_match) => exact_match,
            Err(insertion_point) => {
                if insertion_point == 0 {
                    return None; // Offset is out of bounds (before start)
                }
                insertion_point - 1
            }
        };

        if shard_idx >= self.shards.len() {
            return None;
        }

        // CONFUSING MOMENT: Converting global coordinates back to local shard space.
        // Subtract the shard's starting global offset from our target global offset.
        let local_start = global_byte_offset - self.shard_offsets[shard_idx];
        let shard_byte_len = self.shard_lengths[shard_idx] * 4;

        // Ensure we don't read past the end of this *specific* shard.
        // Note: Target requires a 4-byte lookahead shifted right (local_start + 4),
        // so we must ensure `local_start + batch_byte_len + 4` fits in the shard.
        if local_start + batch_byte_len + 4 <= shard_byte_len {
            let mmap = &self.shards[shard_idx];

            let input = mmap.get(local_start..local_start + batch_byte_len)?;
            let target = mmap.get(local_start + 4..local_start + batch_byte_len + 4)?;

            Some((input, target))
        } else {
            // Straddling across two separate shards is unsupported to avoid allocation/copying.
            None
        }
    }

    /// Creates an iterator yielding chunks cast as `&[u32]`.
    pub fn iter_u32(&self, num_elements: usize) -> ChunkedShardIter<'_, u32> {
        ChunkedShardIter::new(self, num_elements)
    }

    /// Creates an iterator yielding chunks cast as raw bytes `&[u8]`.
    pub fn iter_u8(&self, num_elements: usize) -> ChunkedShardIter<'_, u8> {
        ChunkedShardIter::new(self, num_elements)
    }

    /// Creates an iterator yielding chunks cast as `&[i32]`.
    pub fn iter_i32(&self, num_elements: usize) -> ChunkedShardIter<'_, i32> {
        ChunkedShardIter::new(self, num_elements)
    }
}

/// A generic zero-copy iterator over `PretokenizedDataLoader` shards.
pub struct ChunkedShardIter<'a, T> {
    dataloader: &'a PretokenizedDataLoader,
    current_idx: usize, // Element tracking relative to the *current* active shard
    current_file_idx: usize,
    num_elements: usize,
    _marker: PhantomData<T>,
}

impl<'a, T> ChunkedShardIter<'a, T> {
    fn new(dataloader: &'a PretokenizedDataLoader, num_elements: usize) -> Self {
        Self {
            dataloader,
            current_idx: 0,
            current_file_idx: 0,
            num_elements,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Pod> Iterator for ChunkedShardIter<'a, T> {
    type Item = &'a [T];

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
