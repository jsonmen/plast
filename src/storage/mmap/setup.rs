//! Configuration structures for initializing memory-mapped storage.

use super::advice_set::AdviceSet;
use memmap2::MmapOptions;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

/// A builder-style configuration struct for setting up memory-mapped file storage.
///
/// `MmapSetup` allows you to specify the data files to map, along with custom
/// file opening options, memory mapping options, and OS-level memory advice hints.
///
/// # Examples
///
/// ```rust,ignore
/// use plast::storage::setup::MmapSetup;
/// use plast::storage::advice_set::AdviceSet;
///
/// let setup = MmapSetup::new(vec!["data/train_0.bin", "data/train_1.bin"])
///     .read_write()
///     .with_advice_set(AdviceSet::default());
/// ```
#[derive(Debug)]
pub struct MmapSetup {
    data_files: Vec<PathBuf>,
    open_options: OpenOptions,
    mmap_options: MmapOptions,
    advice_set: AdviceSet,
}

impl MmapSetup {
    /// Creates a new `MmapSetup` with the provided data files and sensible defaults.
    ///
    /// By default:
    /// - Files are opened in read-only mode.
    /// - Huge pages (2MB) are requested via `MmapOptions`.
    /// - The OS advice set defaults to optimized settings for Linux.
    pub fn new<I, P>(data_files: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let data_files = data_files
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        let mut open_options = OpenOptions::new();
        open_options.read(true);

        let mut mmap_options = MmapOptions::new();
        mmap_options.huge(Some(21));

        let advice_set = AdviceSet::default();

        Self {
            data_files,
            open_options,
            mmap_options,
            advice_set,
        }
    }

    /// Configures the setup to open the files in both read and write mode.
    pub fn read_write(mut self) -> Self {
        self.open_options.read(true).write(true);
        self
    }

    /// Overrides the default `OpenOptions`.
    pub fn with_open_options(mut self, open_options: OpenOptions) -> Self {
        self.open_options = open_options;
        self
    }

    /// Overrides the default `MmapOptions`.
    pub fn with_mmap_options(mut self, mmap_options: MmapOptions) -> Self {
        self.mmap_options = mmap_options;
        self
    }

    /// Overrides the default `AdviceSet`.
    pub fn with_advice_set(mut self, advice_set: impl Into<AdviceSet>) -> Self {
        self.advice_set = advice_set.into();
        self
    }

    /// Returns a slice of data file paths.
    pub fn data_files(&self) -> &[PathBuf] {
        &self.data_files
    }

    /// Returns a reference to the `OpenOptions`.
    pub fn open_options(&self) -> &OpenOptions {
        &self.open_options
    }

    /// Returns a reference to the `MmapOptions`.
    pub fn mmap_options(&self) -> &MmapOptions {
        &self.mmap_options
    }

    /// Returns a reference to the `AdviceSet`.
    pub fn advice_set(&self) -> &AdviceSet {
        &self.advice_set
    }
}
