use crate::errors::PretokenizerError;
use crossbeam_channel::bounded;
use polars::datatypes::StringChunked;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Pretokenizes a dataset and writes it out to sized file shards.
///
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
    I: Iterator<Item = StringChunked> + Send + 'static,
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

    // CONFUSING MOMENT: Preventing Channel Deadlocks on Worker Panic
    // If our background execution thread panics, `tx` might not be dropped cleanly,
    // causing `rx.recv()` to hang forever. We wrap processing inside a worker thread
    // that safely drops `tx` when exiting scope, signaling the consumer to stop.
    std::thread::spawn(move || {
        let _tx_guard = tx; // Will drop automatically when this thread terminates
        for shard in dataset {
            let tx_shard = _tx_guard.clone();
            shard.par_iter().for_each_with(tx_shard, |tx, opt_text| {
                if let Some(text) = opt_text
                    && let Ok(encoding) = tokenizer_clone.encode_fast(text, false)
                {
                    let ids = encoding.get_ids().to_vec();
                    let _ = tx.send(ids); // Ignore errors if consumer dropped
                }
            });
        }
    });

    let mut shard_count = 0;
    let mut current_shard_bytes = 0;
    let mut shard_tokens = 0;
    let mut shard_start_time = Instant::now();
    let mut buf: Option<BufWriter<File>> = None;
    let mut current_file_path = PathBuf::new();
    let mut generated_shards = Vec::new();

    while let Ok(payload) = rx.recv() {
        let token_count = payload.len() + 1; // Tokens + 1 EOS marker
        let ids_bytes: &[u8] = bytemuck::cast_slice(&payload);
        let payload_bytes_len = ids_bytes.len() + eos_bytes_vec.len();

        // CONFUSING MOMENT: Shard Rollover Logic
        // Check if we need to initialize a brand new shard file due to space limits or initialization.
        if buf.is_none() || current_shard_bytes >= shard_size_bytes {
            if let Some(mut old_buf) = buf.take() {
                old_buf
                    .flush()
                    .map_err(|e| PretokenizerError::FlushFailed {
                        source: e,
                        path: current_file_path.clone(),
                    })?;

                let elapsed = shard_start_time.elapsed().as_secs_f64();
                log::info!(
                    "Finished shard {}. Tokens: {}, Time: {:.2}s, Throughput: {:.2} tok/s",
                    shard_count - 1,
                    shard_tokens,
                    elapsed,
                    if elapsed > 0.0 {
                        shard_tokens as f64 / elapsed
                    } else {
                        0.0
                    }
                );
            }

            shard_tokens = 0;
            shard_start_time = Instant::now();
            current_file_path =
                save_dir_ref.join(format!("pretokenized_shard_{}.bin", shard_count));

            log::debug!("Creating new shard: {:?}", current_file_path);
            let f = File::create(&current_file_path).map_err(|e| {
                PretokenizerError::ShardCreationFailed {
                    source: e,
                    path: current_file_path.clone(),
                }
            })?;

            generated_shards.push(current_file_path.clone());
            buf = Some(BufWriter::with_capacity(512 * 1024, f));
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
    if let Some(mut final_buf) = buf {
        final_buf
            .flush()
            .map_err(|e| PretokenizerError::FlushFailed {
                source: e,
                path: current_file_path.clone(),
            })?;

        let elapsed = shard_start_time.elapsed().as_secs_f64();
        log::info!(
            "Finished final shard {}. Tokens: {}, Time: {:.2}s, Throughput: {:.2} tok/s",
            shard_count - 1,
            shard_tokens,
            elapsed,
            if elapsed > 0.0 {
                shard_tokens as f64 / elapsed
            } else {
                0.0
            }
        );
    }

    Ok(generated_shards)
}
