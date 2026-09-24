use bytes::Bytes;
use std::ops::Range;

pub trait Storage {
    fn len(&self) -> usize;
    fn clear_state(&mut self) -> ();

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slice_sequential(&mut self, req_len: usize) -> Option<Bytes>;
    fn slice_random(&self, range: Range<usize>) -> Option<Bytes>;
}
