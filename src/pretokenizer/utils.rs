use std::path::{Path, PathBuf};
use std::{fs, io};
pub fn fetch_bin_files<P: AsRef<Path>>(dir: P) -> io::Result<Vec<PathBuf>> {
    let mut arrow_paths = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().is_some_and(|ext| ext == "bin") {
            arrow_paths.push(path);
        }
    }

    Ok(arrow_paths)
}

pub fn fetch_arrow_files<P: AsRef<Path>>(dir: P) -> io::Result<Vec<PathBuf>> {
    let mut arrow_paths = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().is_some_and(|ext| ext == "arrow") {
            arrow_paths.push(path);
        }
    }

    Ok(arrow_paths)
}
/// Scans a directory and returns paths to all files with extensions supported by `ShardLoader`.
///
/// Supported extensions include: `parquet`, `arrow`, `ipc`, `jsonl`, `ndjson`, `csv`, and `txt`.
pub fn fetch_data_files<P: AsRef<Path>>(dir: P) -> io::Result<Vec<PathBuf>> {
    let mut data_paths = Vec::new();

    // All extensions supported by the ShardLoader match statement
    let supported_extensions = ["parquet", "arrow", "ipc", "jsonl", "ndjson", "csv", "txt"];

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            // Check if the file's extension is in the supported list
            let is_supported = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| supported_extensions.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false);

            if is_supported {
                data_paths.push(path);
            }
        }
    }

    Ok(data_paths)
}
