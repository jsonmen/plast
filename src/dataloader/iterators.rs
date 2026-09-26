use super::datatypes::DataloaderType;
use crate::storage::Storage;
use std::marker::PhantomData;

#[derive(Debug, PartialEq, Eq)]
pub struct DataloaderIter<'a, S: Storage, D: DataloaderType> {
    storage: &'a mut S,
    num_elements: usize,
    _marker: PhantomData<D>,
}

impl<'a, S: Storage, D: DataloaderType> DataloaderIter<'a, S, D> {
    pub fn new(storage: &'a mut S, num_elements: usize) -> Self {
        Self {
            storage: storage,
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
