use crate::{
    datatypes::{BurnBytesConverter, BytesConverter, DataloaderType},
    iterators::DataloaderIter,
    storage::Storage,
};
use std::marker::PhantomData;

pub const BYTES_PER_TOKEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataloader<S: Storage, D: DataloaderType> {
    storage: S,
    num_elements: usize,
    _marker: PhantomData<D>,
}

impl<S: Storage, D: DataloaderType> Dataloader<S, D> {
    pub fn new(storage: S, num_elements: usize) -> Self {
        Self {
            storage,
            num_elements,
            _marker: PhantomData,
        }
    }
    pub fn iter(&self) -> DataloaderIter<'_, S, D> {
        DataloaderIter::new(&self.storage, self.num_elements * BYTES_PER_TOKEN)
    }

    pub fn iter_bytes(&self) -> DataloaderIter<'_, S, BytesConverter> {
        DataloaderIter::new(&self.storage, self.num_elements * BYTES_PER_TOKEN)
    }
    pub fn iter_burn_bytes(&self) -> DataloaderIter<'_, S, BurnBytesConverter> {
        DataloaderIter::new(&self.storage, self.num_elements * BYTES_PER_TOKEN)
    }
}
