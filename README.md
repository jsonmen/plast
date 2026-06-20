# Plast ⚡ (Fast Text/Token Dataloader)

Plast is a high-performance Rust data pipeline designed to pretokenize datasets and stream them directly into memory using zero-copy memory mapping (`memmap2`).

I’m still relatively new to Rust, so I am incredibly open to code reviews, feedback, and suggestions! If you see something that can be optimized, please open an issue or a PR! I’d love to learn from it.

### Current Performance Scale

* **Pretokenization Speed:** ~6M tokens/sec (it can chew through FineWeb-Edu's `sample-10B` in roughly 30 minutes).
* **DataLoader Streaming Speed:** Up to **3.10 GiB/s** (effectively saturating the physical read limits of my NVMe SSD).

> ⚠️ **Disclaimer**: This crate is in a very early stage of development. Breaking changes may occur between versions as the internal layout stabilizes.

---

## 🛠️ API Showcase

Here is how you can use Plast to pretokenize a dataset and stream it into an active execution engine loop.

```rust
use plast::{PretokenizedDataLoader, ShardLoader, fetch_arrow_files, pretokenize_dataset};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context_window = 32 * 1024;
    let batch_id = 42;
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let dataset_files = fetch_arrow_files("dataset_dir")?;
    
    // Converted to an owned String to ensure clean lifetime boundaries
    let shardl = ShardLoader::new(dataset_files, "text_column_name".to_string());
    
    let eos_id = tokenizer
        .token_to_id("<|endoftext|>")
        .ok_or("No such token in tokenizer".to_string())?;

    // Tokenize and shatter into 2 GiB maximum binary shards
    let generated_shards = pretokenize_dataset(
        &tokenizer,
        shardl,
        "./pretokenized_shards",
        2 * 1024 * 1024 * 1024, // Shard size in bytes (2 GiB)
        eos_id,                 // EOS Token ID
        4196,                   // Write queue capacity
    )?;

    println!("Generated shards: {:?}", generated_shards);
    let loader = PretokenizedDataLoader::map_data(generated_shards)?;

    // Get contiguous input and target slices shifted by 1 token
    if let Some((input_bytes, target_bytes)) = loader.get_tf_batch_u8(batch_id, context_window) {
        // Zero-copy cast directly to integers for your tensor inputs
        let inputs: &[u32] = bytemuck::cast_slice(input_bytes);
        let targets: &[u32] = bytemuck::cast_slice(target_bytes);

        println!(
            "Ready for transfer to gpu and use in model! Batch size: inputs: {} tokens; targets: {}",
            inputs.len(),
            targets.len()
        );
    }
    
    // Or you can iterate over the data sequentially
    for (i, inputs) in loader.iter_u32(context_window).enumerate() {
        println!(
            "Ready for transfer to gpu and use in model! Batch size: {} tokens",
            inputs.len()
        );
        if i >= 10 {
            break;
        }
    }
    Ok(())
}
```
---

## 🗺️ Roadmap / Todo List

* [ ] **Broader Format Support:** Add native streaming support for common data storage formats like Apache Parquet.
* [ ] **Double-Buffered Pre-loading:** Double-buffer shards directly into RAM. While the GPU is crunching the current memory block, the CPU preloads the next shard in the background to break past physical SSD bottlenecks.
* [ ] **Deep Learning Integration:** Build first-class integrations for the `Burn` framework.

---

## 🤖 Behind the Scenes: AI Usage

Yes, I used AI to help build this project! However, it was used as an active engineering assistant for scaffolding boilerplate and writing unit tests, rather than blind "vibe coding."

I stick to smaller, lightweight models (mostly Gemini Flash on Google's free tier), which means I have to look closely at every line of code to verify it makes sense. It keeps me hands-on and ensures I actually understand the underlying architecture of what I'm building!
