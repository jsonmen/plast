use crate::errors::DataLoaderError;
//use crate::storage::Storage;
use bytes::Bytes;
use crossbeam_channel::{Receiver, bounded};
use memmap2::{Advice, MmapOptions};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};

pub struct BufferStorage {
    rx: Receiver<Bytes>,
    worker_handle: Option<JoinHandle<()>>,
    active_buffer: Option<Bytes>,
    shard_offsets: Vec<usize>,
    total_size: usize,
}
impl BufferStorage {
    pub fn load_data<P: AsRef<Path>>(
        data_files: Vec<P>,
        buffer_size: usize,
    ) -> Result<Self, DataLoaderError> {
        let mut shard_offsets = Vec::with_capacity(data_files.len());
        let mut total_size: usize = 0;

        let path_bufs: Vec<PathBuf> = data_files
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        for path in &path_bufs {
            shard_offsets.push(total_size);
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
            worker_handle: Some(worker_handle),
            active_buffer: None,
            shard_offsets,
            total_size,
        })
    }
}
