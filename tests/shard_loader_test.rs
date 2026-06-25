#[cfg(test)]
mod tests {
    use plast::shard_loader::ShardLoader; // Adjusted to absolute module routing path
    use polars::prelude::*;
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::NamedTempFile;

    // =========================================================================
    // Mock File Creators
    // =========================================================================

    fn create_mock_parquet_file(column_name: &str, values: &[&str]) -> NamedTempFile {
        let mut tmp_file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();
        let series = Series::new(column_name.into(), values);
        let mut df = DataFrame::new_infer_height(vec![series.into()]).unwrap();

        ParquetWriter::new(&mut tmp_file).finish(&mut df).unwrap();

        tmp_file.flush().unwrap();
        tmp_file.seek(SeekFrom::Start(0)).unwrap();
        tmp_file
    }

    fn create_mock_ipc_file(column_name: &str, values: &[&str]) -> NamedTempFile {
        let mut tmp_file = tempfile::Builder::new().suffix(".ipc").tempfile().unwrap();
        let series = Series::new(column_name.into(), values);
        let mut df = DataFrame::new_infer_height(vec![series.into()]).unwrap();

        IpcWriter::new(&mut tmp_file).finish(&mut df).unwrap();

        tmp_file.flush().unwrap();
        tmp_file.seek(SeekFrom::Start(0)).unwrap();
        tmp_file
    }

    fn create_mock_ipc_stream_file(column_name: &str, values: &[&str]) -> NamedTempFile {
        let mut tmp_file = tempfile::Builder::new().suffix(".ipc").tempfile().unwrap();
        let series = Series::new(column_name.into(), values);
        let mut df = DataFrame::new_infer_height(vec![series.into()]).unwrap();

        IpcStreamWriter::new(&mut tmp_file).finish(&mut df).unwrap();

        tmp_file.flush().unwrap();
        tmp_file.seek(SeekFrom::Start(0)).unwrap();
        tmp_file
    }

    fn create_mock_jsonl_file(column_name: &str, values: &[&str]) -> NamedTempFile {
        let mut tmp_file = tempfile::Builder::new()
            .suffix(".jsonl")
            .tempfile()
            .unwrap();
        let series = Series::new(column_name.into(), values);
        let mut df = DataFrame::new_infer_height(vec![series.into()]).unwrap();

        JsonWriter::new(&mut tmp_file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut df)
            .unwrap();

        tmp_file.flush().unwrap();
        tmp_file.seek(SeekFrom::Start(0)).unwrap();
        tmp_file
    }

    fn create_mock_csv_file(column_name: &str, values: &[&str]) -> NamedTempFile {
        let mut tmp_file = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
        let series = Series::new(column_name.into(), values);
        let mut df = DataFrame::new_infer_height(vec![series.into()]).unwrap();

        CsvWriter::new(&mut tmp_file).finish(&mut df).unwrap();

        tmp_file.flush().unwrap();
        tmp_file.seek(SeekFrom::Start(0)).unwrap();
        tmp_file
    }

    // =========================================================================
    // Test Implementations
    // =========================================================================

    #[test]
    fn test_all_file_types_sequential_loading() {
        // Generate one file per format configuration
        let file_parquet = create_mock_parquet_file("text_col", &["parquet_1", "parquet_2"]);
        let file_ipc = create_mock_ipc_file("text_col", &["ipc_1", "ipc_2"]);
        let file_ipc_stream = create_mock_ipc_stream_file("text_col", &["stream_1", "stream_2"]);
        let file_jsonl = create_mock_jsonl_file("text_col", &["jsonl_1", "jsonl_2"]);
        let file_csv = create_mock_csv_file("text_col", &["csv_1", "csv_2"]);

        let files = vec![
            file_parquet.path().to_path_buf(),
            file_ipc.path().to_path_buf(),
            file_ipc_stream.path().to_path_buf(),
            file_jsonl.path().to_path_buf(),
            file_csv.path().to_path_buf(),
        ];

        let mut loader = ShardLoader::new(files, "text_col");

        // 1. Verify Parquet parsing
        let chunk = loader.next().unwrap().unwrap();
        assert_eq!(chunk.len(), 2);
        assert_eq!(chunk.get(0), Some("parquet_1"));

        // 2. Verify Standard IPC random-access parsing
        let chunk = loader.next().unwrap().unwrap();
        assert_eq!(chunk.len(), 2);
        assert_eq!(chunk.get(1), Some("ipc_2"));

        // 3. Verify Fallback IPC Stream sequential parsing
        let chunk = loader.next().unwrap().unwrap();
        assert_eq!(chunk.len(), 2);
        assert_eq!(chunk.get(0), Some("stream_1"));

        // 4. Verify JSONL/NDJSON line scanning
        let chunk = loader.next().unwrap().unwrap();
        assert_eq!(chunk.len(), 2);
        assert_eq!(chunk.get(1), Some("jsonl_2"));

        // 5. Verify CSV comma separated engine scanning
        let chunk = loader.next().unwrap().unwrap();
        assert_eq!(chunk.len(), 2);
        assert_eq!(chunk.get(0), Some("csv_1"));

        // 6. Terminal bounds validation
        assert!(loader.next().is_none());
    }

    #[test]
    fn test_unsupported_file_extension_error() {
        let mut tmp_file = tempfile::Builder::new()
            .suffix(".invalid")
            .tempfile()
            .unwrap();
        writeln!(tmp_file, "dummy content").unwrap();

        let mut loader = ShardLoader::new(vec![tmp_file.path().to_path_buf()], "text_col");
        let result = loader.next().unwrap();

        assert!(matches!(
            result,
            Err(plast::errors::ShardLoaderError::UnsupportedExtension { .. })
        ));
    }

    #[test]
    fn test_missing_column_error() {
        // Create an IPC file containing a mismatched target column signature
        let file = create_mock_ipc_file("wrong_column_name", &["data"]);

        let mut loader = ShardLoader::new(vec![file.path().to_path_buf()], "text_col");
        let result = loader.next().unwrap();

        assert!(matches!(
            result,
            Err(plast::errors::ShardLoaderError::ColumnMissing { .. })
        ));
    }
}
