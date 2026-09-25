use crate::advice_set::AdviceSet;
use memmap2::MmapOptions;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

pub struct MmapSetup {
    data_files: Vec<PathBuf>,
    open_options: OpenOptions,
    mmap_options: MmapOptions,
    advice_set: AdviceSet,
}

impl MmapSetup {
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
    pub fn read_write(mut self) -> Self {
        self.open_options.read(true).write(true);
        self
    }
    pub fn with_open_options(mut self, open_options: OpenOptions) -> Self {
        self.open_options = open_options;
        self
    }
    pub fn with_mmap_options(mut self, mmap_options: MmapOptions) -> Self {
        self.mmap_options = mmap_options;
        self
    }
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
