#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use plast::dataloader::Dataloader;
    use plast::datatypes::{BytesConverter, DataloaderType};
    use plast::mmap_setup::MmapSetup;
    use plast::mmap_storage::MmapStorage;
    use plast::storage::Storage;
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

        let paths = vec![file1.path(), file2.path()];
        let storage = MmapStorage::load_data(MmapSetup::new(paths)).unwrap();

        assert_eq!(storage.total_size(), 10);
        assert_eq!(storage.len(), 10);
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_invalid_byte_alignment_error() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        tmp_file.write_all(&[1u8, 2, 3]).unwrap(); // 3 bytes (Not 4-byte aligned)
        tmp_file.flush().unwrap();

        let paths = vec![tmp_file.path()];
        let result = MmapStorage::load_data(MmapSetup::new(paths));

        assert!(result.is_err());
    }

    // =========================================================================
    // 2. Locate Feature Tests
    // =========================================================================

    #[test]
    fn test_locate_global_to_local_translation() {
        let shard1 = create_mock_shard(&[10, 20, 30, 40]); // 16 bytes (offsets 0..16)
        let shard2 = create_mock_shard(&[50, 60, 70]); // 12 bytes (offsets 16..28)

        let storage =
            MmapStorage::load_data(MmapSetup::new(vec![shard1.path(), shard2.path()])).unwrap();

        // Shard 0 start
        assert_eq!(storage.locate(0), (0, 0));
        // Shard 0 middle
        assert_eq!(storage.locate(8), (0, 8));
        // Shard 1 boundary start (byte offset 16)
        assert_eq!(storage.locate(16), (1, 0));
        // Shard 1 middle (byte offset 20)
        assert_eq!(storage.locate(20), (1, 4));
    }

    // =========================================================================
    // 3. Random Slicing (`slice_random`) Tests
    // =========================================================================

    #[test]
    fn test_slice_random_valid_and_bounds() {
        let shard1_data = vec![100u32, 200, 300, 400]; // 16 bytes
        let file1 = create_mock_shard(&shard1_data);

        let storage = MmapStorage::load_data(MmapSetup::new(vec![file1.path()])).unwrap();

        // Slice first 2 u32 tokens (8 bytes)
        let bytes = storage.slice_random(0..8).unwrap();
        let tokens: &[u32] = bytemuck::cast_slice(&bytes[..]);
        assert_eq!(tokens, &[100, 200]);

        // Slice middle tokens
        let bytes_mid = storage.slice_random(4..12).unwrap();
        let tokens_mid: &[u32] = bytemuck::cast_slice(&bytes_mid[..]);
        assert_eq!(tokens_mid, &[200, 300]);

        // Out-of-bounds range
        assert!(storage.slice_random(0..20).is_none());
        // Zero-length range
        assert!(storage.slice_random(4..4).is_none());
    }

    #[test]
    fn test_slice_random_cross_shard_rejection() {
        let shard1 = create_mock_shard(&[1, 2]); // 8 bytes
        let shard2 = create_mock_shard(&[3, 4]); // 8 bytes

        let storage =
            MmapStorage::load_data(MmapSetup::new(vec![shard1.path(), shard2.path()])).unwrap();

        // Request starting in Shard 0 (byte 4) and extending 8 bytes into Shard 1
        // Straddles boundary (local_offset 4 + req_len 8 > shard 0 length 8)
        assert!(storage.slice_random(4..12).is_none());
    }

    // =========================================================================
    // 4. Sequential Slicing (`slice_sequential`) Tests
    // =========================================================================

    #[test]
    fn test_slice_sequential_rollover() {
        let shard1 = create_mock_shard(&[1, 2, 3, 4]); // 16 bytes
        let shard2 = create_mock_shard(&[5, 6, 7, 8]); // 16 bytes

        let mut storage =
            MmapStorage::load_data(MmapSetup::new(vec![shard1.path(), shard2.path()])).unwrap();

        // Fetch 8-byte chunk from shard 0
        let slice1 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice1[..]), &[1, 2]);

        // Fetch next 8-byte chunk from shard 0
        let slice2 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice2[..]), &[3, 4]);

        // Next fetch exceeds shard 0 -> rolls over to shard 1
        let slice3 = storage.slice_sequential(8).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&slice3[..]), &[5, 6]);
    }

    // =========================================================================
    // 5. High-Level `Dataloader` and Iterators Tests
    // =========================================================================

    #[test]
    fn test_dataloader_iteration() {
        let shard1 = create_mock_shard(&[10, 20, 30, 40]);
        let shard2 = create_mock_shard(&[50, 60, 70, 80]);

        let storage =
            MmapStorage::load_data(MmapSetup::new(vec![shard1.path(), shard2.path()])).unwrap();

        // Request 2 elements per chunk (8 bytes)
        let mut loader = Dataloader::<MmapStorage, BytesConverter>::new(storage, 2);
        let mut iter = loader.iter_bytes();

        let chunk1 = iter.next().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&chunk1[..]), &[10, 20]);

        let chunk2 = iter.next().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&chunk2[..]), &[30, 40]);

        // Rollover to shard 2
        let chunk3 = iter.next().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&chunk3[..]), &[50, 60]);

        let chunk4 = iter.next().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&chunk4[..]), &[70, 80]);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_custom_dataloader_converter() {
        struct U32VecConverter;

        impl DataloaderType for U32VecConverter {
            type Output = Vec<u32>;

            fn convert(bytes: Bytes) -> Self::Output {
                bytemuck::cast_slice(&bytes[..]).to_vec()
            }
        }

        let shard = create_mock_shard(&[1, 2, 3, 4]);
        let storage = MmapStorage::load_data(MmapSetup::new(vec![shard.path()])).unwrap();

        let mut loader = Dataloader::<MmapStorage, U32VecConverter>::new(storage, 2);
        let mut iter = loader.iter();

        let res: Vec<u32> = iter.next().unwrap();
        assert_eq!(res, vec![1, 2]);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn test_burn_bytes_converter() {
        use plast::datatypes::BurnBytesConverter;

        let shard = create_mock_shard(&[10, 20, 30, 40]);
        let storage = MmapStorage::load_data(MmapSetup::new(vec![shard.path()])).unwrap();

        let mut loader = Dataloader::<MmapStorage, BurnBytesConverter>::new(storage, 2);
        let mut iter = loader.iter_burn_bytes();

        let burn_bytes = iter.next().unwrap();
        let out: &[u8] = burn_bytes.as_ref();
        assert_eq!(out, &[10, 0, 0, 0, 20, 0, 0, 0]);
    }
}
