#[cfg(test)]
mod tests {
    use plast::shard_loader::ShardLoader;
    use polars::prelude::*;
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::NamedTempFile;

    // Helper utility to write a mock DataFrame to a local temporary Arrow IPC file

    fn create_mock_ipc_file(column_name: &str, values: &[&str]) -> NamedTempFile {
        let mut tmp_file = NamedTempFile::new().unwrap();

        let series = Series::new(column_name.into(), values);
        let mut df = DataFrame::new_infer_height(vec![series.into()]).unwrap();

        // Use a scoped block to guarantee the streaming writer finishes completely
        // and relinquishes its reference before we read it back
        {
            // FORCE the use of the explicit IPC Stream Layout writer
            IpcStreamWriter::new(&mut tmp_file).finish(&mut df).unwrap();
        }

        // Push any remaining runtime buffers directly to disk storage
        tmp_file.flush().unwrap();

        // Rewind file cursor back to the start offset boundary (byte 0)
        tmp_file.seek(SeekFrom::Start(0)).unwrap();

        tmp_file
    }

    #[test]
    fn test_shard_loader_stream_processing() {
        let file1 = create_mock_ipc_file("text_col", &["hello", "rust"]);
        let file2 = create_mock_ipc_file("text_col", &["polars", "streaming"]);

        let files = vec![file1.path().to_path_buf(), file2.path().to_path_buf()];

        // Ensure constructor accepts string slices natively via `Into<String>` traits
        let mut loader = ShardLoader::new(files, "text_col");

        // Verify shard 1 contents
        let chunk1 = loader.next().unwrap();
        assert_eq!(chunk1.len(), 2);
        assert_eq!(chunk1.get(0), Some("hello"));

        // Verify shard 2 contents
        let chunk2 = loader.next().unwrap();
        assert_eq!(chunk2.len(), 2);
        assert_eq!(chunk2.get(1), Some("streaming"));

        // Verify terminal iterator boundary is reached cleanly
        assert!(loader.next().is_none());
    }
}
