use crate::errors::DataLoaderError;
use crate::storage::Storage;
use bytes::Bytes;
use crossbeam_channel::{Receiver, bounded};
use memmap2::{Advice, MmapOptions};
use std::fs::{self, File};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};

pub struct BufferStorage {
    rx: Receiver<Bytes>,
    _worker_handle: Option<JoinHandle<()>>,
    active_buffer: Option<Bytes>,
    data_files: Vec<PathBuf>,
    total_shard_count: usize,
    current_shard_idx: usize,
    local_cursor: usize,
    total_size: usize,
}
impl BufferStorage {
    pub fn load_data<P: AsRef<Path>>(
        data_files: Vec<P>,
        buffer_size: usize,
    ) -> Result<Self, DataLoaderError> {
        let mut total_size: usize = 0;
        let total_shard_count = data_files.len();

        let path_bufs: Vec<PathBuf> = data_files
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        let data_files = path_bufs.clone();

        for path in &path_bufs {
            let file_len = fs::metadata(path)
                .map_err(|e| DataLoaderError::ShardOpenFailed {
                    source: e,
                    path: path.clone(),
                })?
                .len() as usize;
            total_size += file_len;
        }

        let (tx, rx) = bounded(buffer_size);

        let worker_handle = thread::spawn(move || {
            for path in path_bufs {
                let file = File::open(&path).unwrap();
                let mmap = unsafe { MmapOptions::new().populate().map(&file).unwrap() };
                let _ = mmap.advise(Advice::WillNeed);

                let bytes = Bytes::from_owner(mmap);

                if tx.send(bytes).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            rx,
            _worker_handle: Some(worker_handle),
            active_buffer: None,
            data_files,
            total_shard_count,
            current_shard_idx: 0,
            local_cursor: 0,
            total_size,
        })
    }
}

impl Storage for BufferStorage {
    fn len(&self) -> usize {
        self.total_size
    }
    fn clear_state(&mut self) -> () {
        let buffer_size = self.rx.capacity().unwrap(); // Safe because channel is always bounded
        drop(std::mem::replace(&mut self.rx, crossbeam_channel::never()));

        if let Some(handle) = self._worker_handle.take() {
            let _ = handle.join();
        }

        let (tx, rx) = bounded(buffer_size);
        let data_files = self.data_files.clone();

        let worker_handle = thread::spawn(move || {
            for path in data_files {
                let Ok(file) = File::open(&path) else {
                    continue;
                };
                let Ok(mmap) = (unsafe { MmapOptions::new().populate().map(&file) }) else {
                    continue;
                };
                let _ = mmap.advise(Advice::WillNeed);

                let bytes = Bytes::from_owner(mmap);

                if tx.send(bytes).is_err() {
                    break;
                }
            }
        });

        self.rx = rx;
        self._worker_handle = Some(worker_handle);
        self.current_shard_idx = 0;
        self.local_cursor = 0;
    }
    fn slice_sequential(&mut self, req_len: usize) -> Option<Bytes> {
        while self.current_shard_idx <= self.total_shard_count {
            if let Some(active_buffer) = &self.active_buffer {
                if self.local_cursor + req_len <= active_buffer.len() {
                    let start = self.local_cursor;
                    self.local_cursor += req_len;
                    return Some(active_buffer.slice(start..start + req_len));
                }
            }
            if let Ok(next_buffer) = self.rx.recv() {
                self.active_buffer = Some(next_buffer);
                self.current_shard_idx += 1;
                self.local_cursor = 0;
            } else {
                return None;
            }
        }
        None
    }
    fn slice_random(&self, _range: Range<usize>) -> Option<Bytes> {
        None // Can't be implemented
    }
}
