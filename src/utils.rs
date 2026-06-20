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
