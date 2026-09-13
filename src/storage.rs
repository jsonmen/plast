use bytes::Bytes;
use std::ops::Range;

pub trait Storage {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slice_sequential(
        &self,
        req_len: usize,
        current_shard_idx: &mut usize,
        local_cursor: &mut usize,
    ) -> Option<Bytes>;
    fn slice_random(&self, range: Range<usize>) -> Option<Bytes>;
}
