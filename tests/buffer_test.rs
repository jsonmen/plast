#[cfg(test)]
mod tests {
    use plast::{BufferStorage, Storage};
    use std::io::Write;
    use tempfile::NamedTempFile;
    /// Helper function to create temporary valid mock files filled with 4-byte tokens.
    fn create_mock_shard(data: &[u32]) -> NamedTempFile {
        let mut tmp_file = NamedTempFile::new().unwrap();
        let bytes = bytemuck::cast_slice(data);
        tmp_file.write_all(bytes).unwrap();
        tmp_file.flush().unwrap();
        tmp_file
    }

    // =========================================================================
    // 1. Storage & Initialization Tests
    // =========================================================================

    #[test]
    fn test_successful_mapping_and_invariants() {
        let shard1_data = vec![1u32, 2, 3, 4]; // 16 bytes
        let shard2_data = vec![5u32, 6, 7, 8, 9, 10]; // 24 bytes

        let file1 = create_mock_shard(&shard1_data);
        let file2 = create_mock_shard(&shard2_data);

        let paths = vec![file1.path().to_path_buf(), file2.path().to_path_buf()];
        let storage = BufferStorage::load_data(paths, 2).unwrap();

        assert_eq!(storage.len(), 40); // 16 + 24 bytes
    }

    #[test]
    fn test_empty_file_handling() {
        let tmp_file = NamedTempFile::new().unwrap(); // 0 bytes
        let mut storage = BufferStorage::load_data(vec![tmp_file.path()], 2).unwrap();

        assert_eq!(storage.len(), 0);
        assert!(storage.slice_sequential(4).is_none());
    }

    // =========================================================================
    // 2. Sequential Slicing (`slice_sequential`) Tests
    // =========================================================================

    #[test]
    fn test_slice_sequential_basic() {
        let shard1_data = vec![1u32, 2, 3, 4]; // 16 bytes
        let file1 = create_mock_shard(&shard1_data);

        let mut storage = BufferStorage::load_data(vec![file1.path()], 2).unwrap();

        let slice1 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice1[..]), &[1, 2]);

        let slice2 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice2[..]), &[3, 4]);

        // Exceeds total size
        assert!(storage.slice_sequential(8).is_none());
    }

    #[test]
    fn test_slice_sequential_rollover() {
        let shard1 = create_mock_shard(&[1, 2, 3, 4]); // 16 bytes
        let shard2 = create_mock_shard(&[5, 6, 7, 8]); // 16 bytes

        let mut storage = BufferStorage::load_data(vec![shard1.path(), shard2.path()], 2).unwrap();

        // Fetch 8-byte chunk from shard 0
        let slice1 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice1[..]), &[1, 2]);

        // Fetch next 8-byte chunk from shard 0
        let slice2 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice2[..]), &[3, 4]);

        // Next fetch exceeds shard 0 remaining capacity -> rolls over to shard 1
        let slice3 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice3[..]), &[5, 6]);
    }

    #[test]
    fn test_slice_sequential_cross_shard_boundary_skips_remainder() {
        // This test validates the specific behavior of the current implementation:
        // If a request doesn't fit in the remainder of the current shard,
        // the implementation fetches the next shard and resets the cursor,
        // effectively SKIPPING the remaining bytes of the previous shard.
        let shard1 = create_mock_shard(&[1, 2, 3, 4]); // 16 bytes
        let shard2 = create_mock_shard(&[5, 6, 7, 8]); // 16 bytes

        let mut storage = BufferStorage::load_data(vec![shard1.path(), shard2.path()], 2).unwrap();

        // Read 12 bytes (3 tokens) from shard 1
        let slice1 = storage.slice_sequential(12).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice1[..]), &[1, 2, 3]);

        // Now local_cursor is 12. Request 8 bytes.
        // 12 + 8 = 20 > 16 (shard1 length).
        // The implementation fetches shard2, resets local_cursor to 0,
        // and returns the first 8 bytes of shard2. Token '4' is skipped.
        let slice2 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice2[..]), &[5, 6]);
    }

    // =========================================================================
    // 3. Random Slicing (`slice_random`) Tests
    // =========================================================================

    #[test]
    fn test_slice_random_returns_none() {
        let shard1 = create_mock_shard(&[1, 2, 3, 4]);
        let storage = BufferStorage::load_data(vec![shard1.path()], 2).unwrap();

        // By design, BufferStorage cannot support random access
        assert!(storage.slice_random(0..8).is_none());
        assert!(storage.slice_random(10..20).is_none());
    }

    #[test]
    fn test_slice_tf_sequential_basic() {
        // 5 tokens = 20 bytes.
        // Requesting 8 bytes (2 tokens) means we need 8 + 4 = 12 bytes total per call.
        let shard = create_mock_shard(&[1, 2, 3, 4, 5]);
        let mut storage = BufferStorage::load_data(vec![shard.path()], 2).unwrap();

        // First call: reads bytes 0..12 (tokens 1, 2, 3)
        let (input1, target1) = storage.slice_tf_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&input1[..]), &[1, 2]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&target1[..]), &[2, 3]);

        // Second call: cursor advanced by req_len (8 bytes), so it starts at byte 8 (token 3).
        // Reads bytes 8..20 (tokens 3, 4, 5)
        let (input2, target2) = storage.slice_tf_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&input2[..]), &[3, 4]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&target2[..]), &[4, 5]);

        // Third call: cursor at byte 16. Needs 12 bytes, but only 4 bytes remain. Returns None.
        assert!(storage.slice_tf_sequential(8).is_none());
    }

    #[test]
    fn test_slice_tf_random_returns_none() {
        let shard = create_mock_shard(&[1, 2, 3, 4]);
        let storage = BufferStorage::load_data(vec![shard.path()], 2).unwrap();

        // By design, BufferStorage does not support random access,
        // including Teacher Forcing random access.
        assert!(storage.slice_tf_random(0..8).is_none());
        assert!(storage.slice_tf_random(10..20).is_none());
    }
}
