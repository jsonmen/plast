pub mod error;

#[cfg(feature = "experimental-buffer-storage")]
pub mod buffer;

#[cfg(feature = "mmap-storage")]
pub mod mmap;

pub use error::DataLoaderError;

#[cfg(feature = "experimental-buffer-storage")]
pub use buffer::BufferStorage;

#[cfg(feature = "mmap-storage")]
pub use mmap::{AdviceSet, MmapSetup, MmapStorage};

use bytes::Bytes;
use std::ops::Range;

pub trait Storage {
    fn len(&self) -> usize;
    fn clear_state(&mut self);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slice_sequential(&mut self, req_len: usize) -> Option<Bytes>;
    fn slice_random(&self, range: Range<usize>) -> Option<Bytes>;
}
