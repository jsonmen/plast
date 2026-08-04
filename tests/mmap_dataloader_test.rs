#[cfg(test)]
mod tests {
    use plast::mmap_dataloader::MmapPretokenizedDataLoader;
    use std::io::Write;
    use tempfile::NamedTempFile;
    // Helper function to create temporary valid mock files filled with structured tokens
    fn create_mock_shard(data: &[u32]) -> NamedTempFile {
        let mut tmp_file = NamedTempFile::new().unwrap();
        let bytes = bytemuck::cast_slice(data);
        tmp_file.write_all(bytes).unwrap();
        tmp_file.flush().unwrap();
        tmp_file
    }

    #[test]
    fn test_successful_mapping_and_invariants() {
        let shard1_data = vec![1u32, 2, 3, 4]; // 16 bytes
        let shard2_data = vec![5u32, 6, 7, 8, 9, 10]; // 24 bytes

        let file1 = create_mock_shard(&shard1_data);
        let file2 = create_mock_shard(&shard2_data);

        let paths = vec![file1.path().to_path_buf(), file2.path().to_path_buf()];
        let loader = MmapPretokenizedDataLoader::map_data(paths).unwrap();

        // Validate Structural Metadata assertions
        assert_eq!(loader.total_num_shards(), 2);
        assert_eq!(loader.total_size(), 10); // 4 elements + 6 elements
        assert_eq!(loader.shard_lengths(), &[4, 6]);
        assert_eq!(loader.shard_offsets(), &[0, 16, 40]); // Cumulative byte lengths
    }

    #[test]
    fn test_invalid_byte_alignment_error() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        tmp_file.write_all(&[1u8, 2, 3]).unwrap(); // 3 bytes (Not 4-byte aligned!)
        tmp_file.flush().unwrap();

        let paths = vec![tmp_file.path().to_path_buf()];
        let result = MmapPretokenizedDataLoader::map_data(paths);

        assert!(result.is_err());
    }

    #[test]
    fn test_batch_retrieval_and_shifting() {
        let shard_data = vec![10u32, 20, 30, 40, 50];
        let file = create_mock_shard(&shard_data);

        let loader = MmapPretokenizedDataLoader::map_data(vec![file.path()]).unwrap();

        // Get batch context window size of 2 elements (8 bytes), index 0
        let (input, target) = loader.get_tf_batch_u8(0, 2).unwrap();

        let input_elements: &[u32] = bytemuck::cast_slice(input);
        let target_elements: &[u32] = bytemuck::cast_slice(target);

        // Target should be shifted by exactly 1 element (4 bytes)
        assert_eq!(input_elements, &[10, 20]);
        assert_eq!(target_elements, &[20, 30]);

        // Out-of-bounds index query should gracefully return None
        assert!(loader.get_tf_batch_u8(10, 2).is_none());
    }

    #[test]
    fn test_unified_generic_iterators() {
        let shard1_data = vec![1u32, 2, 3, 4];
        let shard2_data = vec![5u32, 6, 7, 8];

        let file1 = create_mock_shard(&shard1_data);
        let file2 = create_mock_shard(&shard2_data);

        let loader =
            MmapPretokenizedDataLoader::map_data(vec![file1.path(), file2.path()]).unwrap();

        // Test U32 chunk parsing loop
        let mut iter_u32 = loader.iter_u32(2);
        assert_eq!(iter_u32.next(), Some(&[1u32, 2][..]));
        assert_eq!(iter_u32.next(), Some(&[3, 4][..]));
        assert_eq!(iter_u32.next(), Some(&[5, 6][..]));
        assert_eq!(iter_u32.next(), Some(&[7, 8][..]));
        assert_eq!(iter_u32.next(), None);

        // Test identical state engine running casting transformations over I32 type boundaries
        let mut iter_i32 = loader.iter_i32(4);
        assert_eq!(iter_i32.next(), Some(&[1i32, 2, 3, 4][..]));
        assert_eq!(iter_i32.next(), Some(&[5, 6, 7, 8][..]));
        assert_eq!(iter_i32.next(), None);
    }
}
