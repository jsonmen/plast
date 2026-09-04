use crate::errors::ShardLoaderError;
use polars::prelude::*;
use std::path::{Path, PathBuf};

/// A streaming shard loader that reads data files sequentially,
/// extracting a specified text column as a Polars `StringChunked` array.
#[derive(Debug)]
pub struct ShardLoader {
    dataset_files: Vec<PathBuf>,
    text_column_name: String,
    current_file_idx: usize,
}

impl ShardLoader {
    /// Creates a new `ShardLoader`.
    pub fn new<S: Into<String>>(dataset_files: Vec<PathBuf>, text_column_name: S) -> Self {
        Self {
            dataset_files,
            text_column_name: text_column_name.into(),
            current_file_idx: 0,
        }
    }

    fn get_chunked_array<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<StringChunked, ShardLoaderError> {
        let path_ref = path.as_ref();
        let path_buf = path_ref.to_path_buf();

        let extension = path_ref
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let pl_path =
            PlRefPath::try_from_path(path_ref).map_err(|_| ShardLoaderError::InvalidPath {
                path: path_buf.clone(),
            })?;

        // 1. Resolve the initial LazyFrame stage based on extension
        let lazy_frame =
            match extension.as_deref() {
                Some("parquet") => LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())
                    .map_err(|source| ShardLoaderError::PolarsParseFailed {
                        source,
                        path: path_buf.clone(),
                    })?,
                Some("arrow") | Some("ipc") => {
                    // Initialize the Scan operation
                    let scan_result = LazyFrame::scan_ipc(
                        pl_path.clone(),
                        IpcScanOptions::default(),
                        UnifiedScanArgs::default(),
                    );

                    match scan_result {
                        Ok(mut lf) => {
                            // Force validation immediately by extracting the schema metadata
                            if let Err(source) = lf.collect_schema() {
                                let error_msg = source.to_string();
                                if error_msg.contains("InvalidFooter")
                                    || error_msg.contains("out-of-spec")
                                {
                                    // Fallback: Attempt to parse as an IPC Stream
                                    let file = std::fs::File::open(path_ref).map_err(|source| {
                                        ShardLoaderError::FileOpenFailed {
                                            source,
                                            path: path_buf.clone(),
                                        }
                                    })?;

                                    IpcStreamReader::new(file)
                                        .finish()
                                        .map_err(|err| ShardLoaderError::PolarsParseFailed {
                                            source: err,
                                            path: path_buf.clone(),
                                        })?
                                        .lazy()
                                } else {
                                    return Err(ShardLoaderError::PolarsParseFailed {
                                        source,
                                        path: path_buf.clone(),
                                    });
                                }
                            } else {
                                lf
                            }
                        }
                        Err(source) => {
                            return Err(ShardLoaderError::PolarsParseFailed {
                                source,
                                path: path_buf.clone(),
                            });
                        }
                    }
                }
                Some("jsonl") | Some("ndjson") => LazyJsonLineReader::new(pl_path)
                    .finish()
                    .map_err(|source| ShardLoaderError::PolarsParseFailed {
                        source,
                        path: path_buf.clone(),
                    })?,
                Some("csv") | Some("txt") => {
                    LazyCsvReader::new(pl_path).finish().map_err(|source| {
                        ShardLoaderError::PolarsParseFailed {
                            source,
                            path: path_buf.clone(),
                        }
                    })?
                }
                _ => return Err(ShardLoaderError::UnsupportedExtension { path: path_buf }),
            };

        // 2. Select, collect, and extract string components uniformly
        let df = lazy_frame
            .select([col(&self.text_column_name)])
            .collect()
            .map_err(|source| {
                // If Polars failed because the column wasn't found, map it to ColumnMissing
                let err_msg = source.to_string();
                if err_msg.contains("ColumnNotFound") || err_msg.contains("not found") {
                    ShardLoaderError::ColumnMissing {
                        column: self.text_column_name.clone(),
                        path: path_buf.clone(),
                    }
                } else {
                    ShardLoaderError::PolarsParseFailed {
                        source,
                        path: path_buf.clone(),
                    }
                }
            })?;

        let series =
            df.column(&self.text_column_name)
                .map_err(|_| ShardLoaderError::ColumnMissing {
                    column: self.text_column_name.clone(),
                    path: path_buf.clone(),
                })?;

        let chunked_array = series
            .str()
            .map_err(|_| ShardLoaderError::InvalidColumnType {
                column: self.text_column_name.clone(),
                path: path_buf,
            })?;

        Ok(chunked_array.clone())
    }
}

impl Iterator for ShardLoader {
    type Item = Result<StringChunked, ShardLoaderError>;

    /// Advances the iterator, returning the next parsed `StringChunked` array shard or an error.
    fn next(&mut self) -> Option<Self::Item> {
        // 1. Terminal boundary check
        if self.current_file_idx >= self.dataset_files.len() {
            return None;
        }

        // 2. Fetch current file reference and safely increment state pointer.
        let file_path = &self.dataset_files[self.current_file_idx];
        self.current_file_idx += 1;

        // 3. Process data parsing pipeline
        match self.get_chunked_array(file_path) {
            Ok(chunked) => Some(Ok(chunked)),
            Err(err) => Some(Err(err)),
        }
    }
}
