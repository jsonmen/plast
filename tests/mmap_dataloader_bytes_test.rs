#[cfg(test)]
mod tests {
    use plast::errors::DataLoaderError;
    use plast::mmap_dataloader_bytes::MmapPretokenizedDataLoaderBytes;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper function to create temporary valid mock files filled with structured 4-byte tokens
    fn create_mock_shard(data: &[u32]) -> NamedTempFile {
        let mut tmp_file = NamedTempFile::new().unwrap();
        let bytes = bytemuck::cast_slice(data);
        tmp_file.write_all(bytes).unwrap();
        tmp_file.flush().unwrap();
        tmp_file
    }

    #[test]
    fn test_successful_mapping_and_invariants() {
        let shard1_data = vec![1u32, 2, 3, 4]; // 16 bytes, 4 elements
        let shard2_data = vec![5u32, 6, 7, 8, 9, 10]; // 24 bytes, 6 elements

        let file1 = create_mock_shard(&shard1_data);
        let file2 = create_mock_shard(&shard2_data);

        let paths = vec![file1.path().to_path_buf(), file2.path().to_path_buf()];
        let loader = MmapPretokenizedDataLoaderBytes::map_data(paths).unwrap();

        // Validate Structural Metadata assertions
        assert_eq!(loader.total_num_shards(), 2);
        assert_eq!(loader.total_size(), 10); // 4 elements + 6 elements
        assert_eq!(loader.shard_lengths(), &[4, 6]);
        // Cumulative byte lengths: 0, 16 (shard1), 16 + 24 = 40 (shard1 + shard2)
        assert_eq!(loader.shard_offsets(), &[0, 16, 40]);
    }

    #[test]
    fn test_invalid_byte_alignment_error() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        tmp_file.write_all(&[1u8, 2, 3]).unwrap(); // 3 bytes (Not 4-byte aligned!)
        tmp_file.flush().unwrap();

        let paths = vec![tmp_file.path().to_path_buf()];
        let result = MmapPretokenizedDataLoaderBytes::map_data(paths);

        assert!(result.is_err());
        let err = result.unwrap_err();
        // Ensure it's the specific alignment error (adjust match if DataLoaderError enum varies)
        match err {
            DataLoaderError::InvalidByteAlignment { size, path: _ } => {
                assert_eq!(size, 3);
            }
            _ => panic!("Expected InvalidByteAlignment error"),
        }
    }

    #[test]
    fn test_iter_tf_batch_retrieval_and_shifting() {
        // 6 elements = 24 bytes. Allows exactly two batches of 2 elements + 1 shift,
        // leaving no remainder.
        let shard_data = vec![10u32, 20, 30, 40, 50, 60];
        let file = create_mock_shard(&shard_data);

        let loader = MmapPretokenizedDataLoaderBytes::map_data(vec![file.path()]).unwrap();

        let mut iter = loader.iter_tf(2);

        // Batch 1
        let (input, target) = iter.next().unwrap();
        let input_elements: &[u32] = bytemuck::cast_slice(input.as_ref());
        let target_elements: &[u32] = bytemuck::cast_slice(target.as_ref());
        assert_eq!(input_elements, &[10, 20]);
        assert_eq!(target_elements, &[20, 30]);

        // Batch 2
        let (input, target) = iter.next().unwrap();
        let input_elements: &[u32] = bytemuck::cast_slice(input.as_ref());
        let target_elements: &[u32] = bytemuck::cast_slice(target.as_ref());
        assert_eq!(input_elements, &[30, 40]);
        assert_eq!(target_elements, &[40, 50]);

        // Batch 3: Only 1 element (60) remains, which cannot satisfy `num_elements` (2) + shift (1).
        // It should be dropped and return None.
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_iter_chunked_retrieval() {
        let shard1_data = vec![1u32, 2, 3, 4];
        let shard2_data = vec![5u32, 6, 7, 8];

        let file1 = create_mock_shard(&shard1_data);
        let file2 = create_mock_shard(&shard2_data);

        let loader = MmapPretokenizedDataLoaderBytes::map_data(vec![
            file1.path().to_path_buf(),
            file2.path().to_path_buf(),
        ])
        .unwrap();

        let mut iter = loader.iter(2);

        // Helper to compare returned &[u8] with expected u32 slices
        let assert_bytes_eq = |actual: &[u8], expected: &[u32]| {
            assert_eq!(actual, bytemuck::cast_slice::<u32, u8>(expected));
        };

        assert_bytes_eq(iter.next().unwrap(), &[1, 2]);
        assert_bytes_eq(iter.next().unwrap(), &[3, 4]);
        // Rolls over to shard 2 cleanly
        assert_bytes_eq(iter.next().unwrap(), &[5, 6]);
        assert_bytes_eq(iter.next().unwrap(), &[7, 8]);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_iter_drops_straddling_remainder() {
        // Explicitly test the "drop remainder chunk" logic when a window straddles a shard boundary
        let shard1_data = vec![1u32, 2, 3]; // 3 elements (12 bytes)
        let shard2_data = vec![4u32, 5, 6]; // 3 elements (12 bytes)

        let file1 = create_mock_shard(&shard1_data);
        let file2 = create_mock_shard(&shard2_data);

        let loader = MmapPretokenizedDataLoaderBytes::map_data(vec![
            file1.path().to_path_buf(),
            file2.path().to_path_buf(),
        ])
        .unwrap();

        let mut iter = loader.iter(2);

        let assert_bytes_eq = |actual: &[u8], expected: &[u32]| {
            assert_eq!(actual, bytemuck::cast_slice::<u32, u8>(expected));
        };

        // Fits in shard 1
        assert_bytes_eq(iter.next().unwrap(), &[1, 2]);

        // Remaining in shard 1 is just `[3]` (1 element). Requested 2.
        // Should drop `[3]` and advance to shard 2.
        assert_bytes_eq(iter.next().unwrap(), &[4, 5]);

        // Remaining in shard 2 is just `[6]` (1 element). Requested 2.
        // Should drop `[6]` and advance to end.
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_empty_shard_handling() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        tmp_file.flush().unwrap(); // 0 bytes, which is a multiple of 4

        let loader = MmapPretokenizedDataLoaderBytes::map_data(vec![tmp_file.path()]).unwrap();

        assert_eq!(loader.total_size(), 0);
        assert_eq!(loader.total_num_shards(), 1);
        assert_eq!(loader.shard_lengths(), &[0]);
        assert_eq!(loader.shard_offsets(), &[0, 0]);

        let mut iter = loader.iter(2);
        assert!(iter.next().is_none());
    }
}
