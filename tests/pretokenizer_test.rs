#[cfg(test)]
mod tests {
    use plast::pretokenizer::pretokenize_dataset;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;
    use tokenizers::Tokenizer;

    fn create_mock_tokenizer() -> Tokenizer {
        // Look up the file relative to this test file's directory
        let large_json_data = include_str!("../fixtures/mock_tokenizer.json");

        let mut tmp_json_file = NamedTempFile::new().expect("Failed to create temp file");
        tmp_json_file
            .write_all(large_json_data.as_bytes())
            .expect("Failed to write mock data");
        tmp_json_file.flush().expect("Failed to flush file buffers");

        Tokenizer::from_file(tmp_json_file.path()).expect("Failed to load tokenizer")
    }

    #[test]
    fn test_directory_creation_failure() {
        let tokenizer = create_mock_tokenizer();
        let dataset = vec![].into_iter();

        // Pass a completely invalid out-of-bounds target filename directory path
        let invalid_path = Path::new("/sys/class/non_existent_directory_target/data");

        let result = pretokenize_dataset(&tokenizer, dataset, invalid_path, 1024, 0, 5);

        assert!(result.is_err());
    }
}
