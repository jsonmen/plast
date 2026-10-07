//! Iterators for traversing data managed by the dataloader.

use super::datatypes::DataloaderType;
use crate::storage::Storage;
use std::marker::PhantomData;

/// An iterator that yields elements of type `D::Output` from a given storage backend.
///
/// This struct is typically created via the [`Dataloader::iter`](crate::dataloader::loader::Dataloader::iter) method and handles
/// sequentially pulling byte chunks from the storage and converting them on the fly.
///
/// # Examples
///
/// ```rust,ignore
/// use plast::dataloader::loader::Dataloader;
/// use plast::dataloader::datatypes::BytesConverter;
/// use plast::storage::MmapStorage;
/// use plast::storage::setup::MmapSetup;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let setup = MmapSetup::new(vec!["data.bin"]);
/// let storage = MmapStorage::load_data(setup)?;
/// let mut loader = Dataloader::<_, BytesConverter>::new(storage, 1024);
///
/// for chunk in loader.iter() {
///     // `chunk` is of type `Bytes`
///     println!("Read chunk of size: {} bytes", chunk.len());
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct DataloaderIter<'a, S: Storage, D: DataloaderType> {
    storage: &'a mut S,
    num_elements: usize,
    _marker: PhantomData<D>,
}

impl<'a, S: Storage, D: DataloaderType> DataloaderIter<'a, S, D> {
    /// Creates a new `DataloaderIter`.
    ///
    /// # Arguments
    /// * `storage` - A mutable reference to the underlying storage.
    /// * `num_elements` - The number of bytes to read per iteration.
    pub fn new(storage: &'a mut S, num_elements: usize) -> Self {
        Self {
            storage,
            num_elements,
            _marker: PhantomData,
        }
    }
}

impl<'a, S: Storage, D: DataloaderType> Iterator for DataloaderIter<'a, S, D> {
    type Item = D::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bytes) = self.storage.slice_sequential(self.num_elements) {
            return Some(D::convert(bytes));
        }
        None
    }
}

/// An iterator that yields tuples of `(D::Output, D::Output)` (input and target)
/// from a given storage backend for teacher forcing.
///
/// This struct is typically created via a teacher-forcing-specific iteration method
/// (e.g., [`Dataloader::tf_iter`](crate::dataloader::loader::Dataloader::tf_iter)) and handles
/// sequentially pulling byte chunks from the storage and converting them on the fly.
/// The target sequence (second element of the tuple) is shifted by one token/element
/// relative to the input sequence (first element).
///
/// # Examples
///
/// ```rust,ignore
/// use plast::dataloader::loader::Dataloader;
/// use plast::dataloader::datatypes::BytesConverter;
/// use plast::storage::MmapStorage;
/// use plast::storage::setup::MmapSetup;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let setup = MmapSetup::new(vec!["data.bin"]);
/// let storage = MmapStorage::load_data(setup)?;
/// let mut loader = Dataloader::<_, BytesConverter>::new(storage, 1024);
///
/// for (input_chunk, target_chunk) in loader.tf_iter() {
///     // `input_chunk` and `target_chunk` are of type `Bytes` (or D::Output)
///     // `target_chunk` is shifted by one token/element relative to `input_chunk`
///     println!("Read input chunk of size: {} bytes", input_chunk.len());
///     println!("Read target chunk of size: {} bytes", target_chunk.len());
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct TFDataloaderIter<'a, S: Storage, D: DataloaderType> {
    storage: &'a mut S,
    num_elements: usize,
    _marker: PhantomData<D>,
}

impl<'a, S: Storage, D: DataloaderType> TFDataloaderIter<'a, S, D> {
    /// Creates a new `TFDataloaderIter`.
    ///
    /// # Arguments
    /// * `storage` - A mutable reference to the underlying storage.
    /// * `num_elements` - The base number of bytes to read per iteration for the input sequence.
    ///   The storage will internally handle reading the extra bytes needed for the shifted target sequence.
    pub fn new(storage: &'a mut S, num_elements: usize) -> Self {
        Self {
            storage,
            num_elements,
            _marker: PhantomData,
        }
    }
}

impl<'a, S: Storage, D: DataloaderType> Iterator for TFDataloaderIter<'a, S, D> {
    type Item = (D::Output, D::Output);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((input_bytes, target_bytes)) =
            self.storage.slice_tf_sequential(self.num_elements)
        {
            return Some((D::convert(input_bytes), D::convert(target_bytes)));
        }
        None
    }
}
