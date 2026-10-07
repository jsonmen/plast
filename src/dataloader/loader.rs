//! Core dataloader orchestration.
//!
//! This module provides the main `Dataloader` struct which wraps a `Storage`
//! implementation and provides convenient iterators for consuming data in batches.

use super::{
    datatypes::{BytesConverter, DataloaderType},
    iterators::{DataloaderIter, TFDataloaderIter},
};
use crate::storage::Storage;
use std::marker::PhantomData;

#[cfg(feature = "burn")]
use super::datatypes::BurnBytesConverter;

/// The number of bytes that constitute a single element (token).
///
/// By default, this assumes elements are 32-bit integers/floats (e.g., `u32`, `i32`, `f32`),
/// meaning each element takes up 4 bytes of storage.
pub const BYTES_PER_TOKEN: usize = 4;

/// The primary orchestrator for reading data from a storage backend in batches.
///
/// `Dataloader` takes ownership of a `Storage` implementation and allows you to
/// iterate over the stored data in fixed-size chunks, automatically converting the
/// raw bytes into your desired output type using a `DataloaderType` implementation.
///
/// # Type Parameters
/// * `S` - The storage backend (e.g., [`MmapStorage`](crate::storage::MmapStorage), [`BufferStorage`](crate::storage::BufferStorage)).
/// * `D` - The data type converter (e.g., [`BytesConverter`], [`BurnBytesConverter`]).
///
/// # Examples
///
/// ## Creating and iterating a dataloader
/// ```rust,ignore
/// use plast::dataloader::loader::Dataloader;
/// use plast::dataloader::datatypes::BytesConverter;
/// use plast::storage::MmapStorage;
/// use plast::storage::setup::MmapSetup;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let setup = MmapSetup::new(vec!["train.bin"]);
/// let storage = MmapStorage::load_data(setup)?;
///
/// // Read batches of 1024 elements (4096 bytes)
/// let mut loader = Dataloader::<_, BytesConverter>::new(storage, 1024);
///
/// for batch in loader.iter() {
///     assert_eq!(batch.len(), 1024 * 4);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataloader<S: Storage, D: DataloaderType> {
    storage: S,
    num_elements: usize,
    _marker: PhantomData<D>,
}

impl<S: Storage, D: DataloaderType> Dataloader<S, D> {
    /// Creates a new `Dataloader`.
    ///
    /// # Arguments
    /// * `storage` - The storage backend to read data from.
    /// * `num_elements` - The number of elements (tokens) to yield per iteration.
    ///   The actual number of bytes read per iteration will be `num_elements * BYTES_PER_TOKEN`.
    pub fn new(storage: S, num_elements: usize) -> Self {
        Self {
            storage,
            num_elements,
            _marker: PhantomData,
        }
    }

    /// Resets the internal cursor of the underlying storage.
    ///
    /// This allows you to re-iterate over the dataset from the beginning
    /// without recreating the `Dataloader`.
    pub fn clear_state(&mut self) {
        self.storage.clear_state();
    }

    /// Returns an iterator that yields `D::Output` over the dataset.
    ///
    /// The iterator will yield batches of `self.num_elements` elements.
    pub fn iter(&mut self) -> DataloaderIter<'_, S, D> {
        DataloaderIter::new(&mut self.storage, self.num_elements * BYTES_PER_TOKEN)
    }

    /// Returns an iterator that yields raw `Bytes` regardless of the configured `D` type.
    pub fn iter_bytes(&mut self) -> DataloaderIter<'_, S, BytesConverter> {
        DataloaderIter::new(&mut self.storage, self.num_elements * BYTES_PER_TOKEN)
    }

    /// Returns an iterator that yields Burn's `Bytes` type.
    ///
    /// Requires the `burn` feature to be enabled.
    #[cfg(feature = "burn")]
    pub fn iter_burn_bytes(&mut self) -> DataloaderIter<'_, S, BurnBytesConverter> {
        DataloaderIter::new(&mut self.storage, self.num_elements * BYTES_PER_TOKEN)
    }
    /// Returns an iterator that yields `D::Output` over the dataset.
    ///
    /// The iterator will yield batches of `self.num_elements` elements.
    pub fn tf_iter(&mut self) -> TFDataloaderIter<'_, S, D> {
        TFDataloaderIter::new(&mut self.storage, self.num_elements * BYTES_PER_TOKEN)
    }

    /// Returns an iterator that yields raw `Bytes` regardless of the configured `D` type.
    pub fn tf_iter_bytes(&mut self) -> TFDataloaderIter<'_, S, BytesConverter> {
        TFDataloaderIter::new(&mut self.storage, self.num_elements * BYTES_PER_TOKEN)
    }

    /// Returns an iterator that yields Burn's `Bytes` type.
    ///
    /// Requires the `burn` feature to be enabled.
    #[cfg(feature = "burn")]
    pub fn tf_iter_burn_bytes(&mut self) -> TFDataloaderIter<'_, S, BurnBytesConverter> {
        TFDataloaderIter::new(&mut self.storage, self.num_elements * BYTES_PER_TOKEN)
    }
}
