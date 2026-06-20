use crate::errors::ShardLoaderError;
use polars::prelude::*;
use std::fs::File;
use std::path::PathBuf;

/// A streaming shard loader that reads Arrow IPC files sequentially,
/// extracting a specified text column as a Polars `StringChunked` array.
pub struct ShardLoader {
    dataset_files: Vec<PathBuf>,
    text_column_name: String,
    current_file_idx: usize,
}

impl ShardLoader {
    /// Creates a new `ShardLoader`.
    ///
    /// Accepts any parameter that can convert into a `String` to maximize usability.
    pub fn new<S: Into<String>>(dataset_files: Vec<PathBuf>, text_column_name: S) -> Self {
        Self {
            dataset_files,
            text_column_name: text_column_name.into(),
            current_file_idx: 0,
        }
    }

    /// Internal method to progress file state indices and parse the underlying data.
    fn load_next_shard(&mut self) -> Result<Option<StringChunked>, ShardLoaderError> {
        if self.current_file_idx >= self.dataset_files.len() {
            return Ok(None);
        }

        let file_path = &self.dataset_files[self.current_file_idx];

        let f = File::open(file_path).map_err(|e| ShardLoaderError::FileOpenFailed {
            source: e,
            path: file_path.clone(),
        })?;

        // CONFUSING MOMENT: Polars IPC Streaming Reader lifetime tracking
        // `IpcStreamReader` consumes the file handle and builds a streaming dataframe.
        // We use `.finish()` to pull the fully parsed memory layouts into scope.
        let df =
            IpcStreamReader::new(f)
                .finish()
                .map_err(|e| ShardLoaderError::IpcParseFailed {
                    source: e,
                    path: file_path.clone(),
                })?;

        let name_series =
            df.column(&self.text_column_name)
                .map_err(|e| ShardLoaderError::ColumnMissing {
                    source: e,
                    path: file_path.clone(),
                })?;

        // CONFUSING MOMENT: Downcasting Series to specialized StringChunked references
        // Polars Series are generic analytical containers. `.str()` downcasts them
        // to concrete underlying Arrow UTF-8 array references.
        let string_chunked = name_series
            .str()
            .map_err(|e| ShardLoaderError::InvalidColumnType {
                source: e,
                path: file_path.clone(),
            })?
            .clone(); // Clones the lightweight array references/pointers, not the actual text bytes.

        self.current_file_idx += 1;
        Ok(Some(string_chunked))
    }
}

impl Iterator for ShardLoader {
    type Item = StringChunked;

    /// Advances the iterator, returning the next parsed `StringChunked` array shard.
    fn next(&mut self) -> Option<Self::Item> {
        self.load_next_shard().unwrap()
    }
}
