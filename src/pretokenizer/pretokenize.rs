use super::error::{PretokenizerError, ShardLoaderError};
use crossbeam_channel::bounded;
use polars::datatypes::StringChunked;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

fn finalize_shard(
    buf: &mut Option<BufWriter<File>>,
    path: &Path,
    index: usize,
    tokens: usize,
    start: Instant,
) -> Result<(), PretokenizerError> {
    if let Some(mut writer) = buf.take() {
        writer.flush().map_err(|e| PretokenizerError::FlushFailed {
            source: e,
            path: path.to_path_buf(),
        })?;

        let elapsed = start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        };
        log::info!(
            "Finished shard {index}. Tokens: {tokens}, Time: {elapsed:.2}s, Throughput: {tok_per_sec:.2} tok/s"
        );
    }
    Ok(())
}

/// Pretokenizes an iterator of string chunks and outputs binary-encoded token shards.
///
/// Output files are written in little-endian `u32` format as `pretokenized_shard_N.bin`.
/// Log verbosity is handled through standard telemetry (`log::info` / `log::debug`).
pub fn pretokenize_dataset<I, P>(
    tokenizer: &Tokenizer,
    dataset: I,
    save_dir: P,
    shard_size_bytes: usize,
    eos_id: u32,
    write_queue_size: usize,
) -> Result<Vec<PathBuf>, PretokenizerError>
where
    I: Iterator<Item = Result<StringChunked, ShardLoaderError>> + Send + 'static,
    P: AsRef<Path>,
{
    let save_dir_ref = save_dir.as_ref();
    std::fs::create_dir_all(save_dir_ref).map_err(|e| {
        PretokenizerError::DirectoryCreationFailed {
            source: e,
            path: save_dir_ref.to_path_buf(),
        }
    })?;

    let (tx, rx) = bounded::<Vec<u32>>(write_queue_size);
    let tokenizer_clone = tokenizer.clone();
    let eos_bytes_vec = eos_id.to_le_bytes();

    // Producer Thread: Tokenizes in parallel using Rayon and streams token buffers to disk writer.
    // The channel `tx` automatically closes when this thread exits (or panics), causing `rx.recv()` to terminate cleanly.
    std::thread::spawn(move || {
        for shard_result in dataset {
            let shard = match shard_result {
                Ok(s) => s,
                Err(err) => {
                    log::warn!("Skipping corrupted dataset shard: {err}");
                    continue;
                }
            };

            // Parallel tokenization across available Rayon threads
            shard
                .par_iter()
                .for_each_with(tx.clone(), |tx_channel, opt_text| {
                    if let Some(text) = opt_text
                        && let Ok(encoding) = tokenizer_clone.encode_fast(text, false)
                    {
                        let _ = tx_channel.send(encoding.get_ids().to_vec());
                    }
                });
        }
        // Drops the original `tx` handle here when scope ends
    });

    let mut shard_count: usize = 0;
    let mut current_shard_bytes = 0;
    let mut shard_tokens = 0;
    let mut shard_start_time = Instant::now();
    let mut buf: Option<BufWriter<File>> = None;
    let mut current_file_path = PathBuf::new();
    let mut generated_shards = Vec::new();

    while let Ok(payload) = rx.recv() {
        let token_count = payload.len() + 1; // Tokens + EOS
        let ids_bytes: &[u8] = bytemuck::cast_slice(&payload);
        let payload_bytes_len = ids_bytes.len() + eos_bytes_vec.len();

        // Rollover to new shard if size limit exceeded or buffer unitialized
        if buf.is_none() || current_shard_bytes >= shard_size_bytes {
            finalize_shard(
                &mut buf,
                &current_file_path,
                shard_count.saturating_sub(1),
                shard_tokens,
                shard_start_time,
            )?;

            shard_tokens = 0;
            shard_start_time = Instant::now();
            current_file_path = save_dir_ref.join(format!("pretokenized_shard_{shard_count}.bin"));

            log::debug!("Creating new shard: {:?}", current_file_path);
            let file = File::create(&current_file_path).map_err(|e| {
                PretokenizerError::ShardCreationFailed {
                    source: e,
                    path: current_file_path.clone(),
                }
            })?;

            generated_shards.push(current_file_path.clone());
            buf = Some(BufWriter::with_capacity(512 * 1024, file));
            current_shard_bytes = 0;
            shard_count += 1;
        }

        if let Some(ref mut buff) = buf {
            buff.write_all(ids_bytes)
                .map_err(|e| PretokenizerError::WriteFailed {
                    source: e,
                    path: current_file_path.clone(),
                })?;
            buff.write_all(&eos_bytes_vec)
                .map_err(|e| PretokenizerError::WriteFailed {
                    source: e,
                    path: current_file_path.clone(),
                })?;

            current_shard_bytes += payload_bytes_len;
            shard_tokens += token_count;
        }
    }

    // Flush the remaining remnants inside the final active buffer
    finalize_shard(
        &mut buf,
        &current_file_path,
        shard_count.saturating_sub(1),
        shard_tokens,
        shard_start_time,
    )?;

    Ok(generated_shards)
}
