use crate::errors::DataLoaderError;
use bytemuck::Pod;
use memmap2::Mmap;
use std::fs::File;
use std::marker::PhantomData;
use std::path::Path;

#[derive(Debug)]
pub struct ShardPretokenizedDataLoader {
    shards: Vec<File>,
    shard_sizes: Vec<usize>,
    shard_cache_size: usize,
    current_inx: usize,
    buffer_one: Vec<u8>,
    buffer_two: Vec<u8>,
    total_size: usize,
}

impl ShardPretokenizedDataLoader {
    pub fn load_data<P: AsRef<Path>>(data_files: Vec<P>) -> Result<Self, DataLoaderError> {
        let files_count = data_files.len();
        let mut shards = Vec::with_capacity(files_count);
        let mut shard_offsets = Vec::with_capacity(files_count + 1);

        let mut total_size = 0;

        for path_ref in data_files {
            let path = path_ref.as_ref();
            let f = File::open(path).map_err(|e| DataLoaderError::ShardOpenFailed {
                source: e,
                path: path.to_path_buf(),
            })?;

            let file_size: u64 = f
                .metadata()
                .map_err(|e| DataLoaderError::ShardOpenFailed {
                    source: e,
                    path: path.to_path_buf(),
                })?
                .len();

            shards.push(f);
            shard_offsets.push(total_size);
            total_size += file_size;
        }
        todo!()
    }
}
