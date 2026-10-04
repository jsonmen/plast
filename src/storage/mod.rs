//! Storage backends and abstractions for the dataloader.
//!
//! This module defines the `Storage` trait which represents an abstract,
//! contiguous block of data that can be sliced sequentially or randomly.
//! It also provides implementations like memory-mapped files and background
//! buffered I/O.

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

/// An abstraction over a data storage backend.
///
/// Implementations of this trait provide mechanisms to read chunks of data
/// either sequentially or via random access ranges. The data is expected to
/// be composed of elements, typically 4 bytes each (e.g., `u32`).
pub trait Storage {
    /// Returns the total number of elements in the storage.
    fn len(&self) -> usize;

    /// Resets any internal cursors or state used for sequential reading.
    fn clear_state(&mut self);

    /// Returns `true` if the storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads `req_len` *bytes* sequentially from the current cursor position.
    ///
    /// If the end of the data is reached before `req_len` bytes can be read,
    /// or if the storage does not support sequential reading, this returns `None`.
    ///
    /// *Note: The parameter `req_len` represents the number of **bytes**, not elements.*
    fn slice_sequential(&mut self, req_len: usize) -> Option<Bytes>;

    /// Reads a slice of *bytes* defined by `range` from the storage.
    ///
    /// If the storage does not support random access, or if the range is out
    /// of bounds, this returns `None`.
    fn slice_random(&self, range: Range<usize>) -> Option<Bytes>;
}
