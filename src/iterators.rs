use crate::{datatypes::DataloaderType, storage::Storage};
use std::marker::PhantomData;

pub struct DataloaderIter<'a, S: Storage, D: DataloaderType> {
    storage: &'a S,
    current_idx: usize,
    current_file_idx: usize,
    num_elements: usize,
    _marker: PhantomData<D>,
}

impl<'a, S: Storage, D: DataloaderType> DataloaderIter<'a, S, D> {
    pub fn new(storage: &'a S, num_elements: usize) -> Self {
        Self {
            storage: storage,
            current_idx: 0,
            current_file_idx: 0,
            num_elements,
            _marker: PhantomData,
        }
    }
}

impl<'a, S: Storage, D: DataloaderType> Iterator for DataloaderIter<'a, S, D> {
    type Item = D::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bytes) = self.storage.slice_sequential(
            self.num_elements,
            &mut self.current_file_idx,
            &mut self.current_idx,
        ) {
            return Some(D::convert(bytes));
        }
        None
    }
}
