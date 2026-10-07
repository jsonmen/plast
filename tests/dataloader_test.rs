#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use plast::{BytesConverter, DataloaderType, MmapSetup, MmapStorage, dataloader::Dataloader};
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
        use plast::BurnBytesConverter;

        let shard = create_mock_shard(&[10, 20, 30, 40]);
        let storage = MmapStorage::load_data(MmapSetup::new(vec![shard.path()])).unwrap();

        let mut loader = Dataloader::<MmapStorage, BurnBytesConverter>::new(storage, 2);
        let mut iter = loader.iter_burn_bytes();

        let burn_bytes = iter.next().unwrap();
        let out: &[u8] = burn_bytes.as_ref();
        assert_eq!(out, &[10, 0, 0, 0, 20, 0, 0, 0]);
    }

    #[test]
    fn test_tf_dataloader_iteration() {
        // 6 tokens = 24 bytes.
        // num_elements = 2 means 8 bytes per step.
        // Each step consumes 8 bytes for the sequence, and needs 4 extra bytes for the target shift.
        // The cursor advances by `req_len` (8 bytes) to yield non-overlapping sequences.
        let shard = create_mock_shard(&[1, 2, 3, 4, 5, 6]);
        let storage = MmapStorage::load_data(MmapSetup::new(vec![shard.path()])).unwrap();

        let mut loader = Dataloader::<MmapStorage, BytesConverter>::new(storage, 2);
        let mut iter = loader.tf_iter_bytes();

        // Chunk 1: starts at byte 0
        let (in1, tgt1) = iter.next().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&in1[..]), &[1, 2]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&tgt1[..]), &[2, 3]);

        // Chunk 2: cursor advances by 8 bytes, starts at byte 8 (token 3)
        let (in2, tgt2) = iter.next().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&in2[..]), &[3, 4]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&tgt2[..]), &[4, 5]);

        // Chunk 3: cursor at byte 16. Needs 12 bytes (16+12=28), but only 24 bytes total. Returns None.
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tf_custom_dataloader_converter() {
        struct U32VecConverter;

        impl DataloaderType for U32VecConverter {
            type Output = Vec<u32>;

            fn convert(bytes: Bytes) -> Self::Output {
                bytemuck::cast_slice(&bytes[..]).to_vec()
            }
        }

        // 5 tokens = 20 bytes. Enough for two non-overlapping chunks of 2 tokens (8 bytes) + 1 target token (4 bytes).
        let shard = create_mock_shard(&[10, 20, 30, 40, 50]);
        let storage = MmapStorage::load_data(MmapSetup::new(vec![shard.path()])).unwrap();

        let mut loader = Dataloader::<MmapStorage, U32VecConverter>::new(storage, 2);
        let mut iter = loader.tf_iter();

        // Chunk 1
        let (input, target) = iter.next().unwrap();
        assert_eq!(input, vec![10, 20]);
        assert_eq!(target, vec![20, 30]);

        // Chunk 2 (advances by req_len = 8 bytes, so starts at token 30)
        let (input2, target2) = iter.next().unwrap();
        assert_eq!(input2, vec![30, 40]);
        assert_eq!(target2, vec![40, 50]);

        // Chunk 3: not enough bytes left (cursor at 16, needs 12 bytes, only 4 remain)
        assert!(iter.next().is_none());
    }

    #[cfg(feature = "burn")]
    #[test]
    fn test_tf_burn_bytes_converter() {
        use plast::BurnBytesConverter;

        let shard = create_mock_shard(&[10, 20, 30]);
        let storage = MmapStorage::load_data(MmapSetup::new(vec![shard.path()])).unwrap();

        let mut loader = Dataloader::<MmapStorage, BurnBytesConverter>::new(storage, 2);
        let mut iter = loader.tf_iter_burn_bytes();

        let (in_burn, tgt_burn) = iter.next().unwrap();
        let in_out: &[u8] = in_burn.as_ref();
        let tgt_out: &[u8] = tgt_burn.as_ref();

        assert_eq!(in_out, &[10, 0, 0, 0, 20, 0, 0, 0]);
        assert_eq!(tgt_out, &[20, 0, 0, 0, 30, 0, 0, 0]);

        assert!(iter.next().is_none());
    }
}
