use crate::errors::DataLoaderError;
use memmap2::{Advice, Mmap, MmapMut};

/// A collection of memory advice hints to be applied to a memory-mapped file.
///
/// This struct allows you to customize how the operating system manages the
/// physical pages of your mapped data. By default, it provides highly optimized
/// settings for Linux servers (`Sequential` + `HugePage`), while remaining safe
/// for macOS and Windows users by defaulting to an empty set.
///
/// # Examples
///
/// ## 1. Using the OS-optimized defaults (Recommended for Linux)
/// ```ignore
/// let storage = MmapStorage::load_data(
///     vec!["data/train.bin"],
///     AdviceSet::default() // Uses Sequential + HugePage on Linux
/// )?;
/// ```
///
/// ## 2. Customizing advice for specific access patterns
/// ```ignore
/// // Use Random access hints if you are shuffling data heavily
/// let storage = MmapStorage::load_data(
///     vec!["data/shuffled.bin"],
///     [Advice::Random, Advice::WillNeed] // Pass an array directly!
/// )?;
/// ```
///
/// ## 3. Cross-platform safety (macOS/Windows)
/// ```ignore
/// // On non-Linux systems, HugePage is unsupported.
/// // Use empty() or basic POSIX flags like Sequential.
/// let storage = MmapStorage::load_data(
///     vec!["data/test.bin"],
///     AdviceSet::empty()
/// )?;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdviceSet {
    advice: Vec<Advice>,
}
impl AdviceSet {
    pub fn new<A: AsRef<[Advice]>>(advices: A) -> Self {
        Self {
            advice: advices.as_ref().to_vec(),
        }
    }
    /// Creates an empty `AdviceSet`.
    ///
    /// This is useful for platforms that do not support `madvise` (like Windows)
    /// or when you want to explicitly disable all memory hints.
    pub fn empty() -> Self {
        Self { advice: vec![] }
    }
    /// Applies the stored advice to the provided memory map.
    ///
    /// Returns a `DataLoaderError::AdviseFailed` if the operating system
    /// rejects any of the provided hints (e.g., using `HugePage` on macOS).
    pub fn apply(&self, mmap: &Mmap) -> Result<(), DataLoaderError> {
        for &advice in &self.advice {
            mmap.advise(advice).map_err(|source| {
                // Map the io::Error to your custom DataLoaderError
                DataLoaderError::AdviseFailed { source, advice }
            })?;
        }
        Ok(())
    }

    pub fn apply_mut(&self, mmap: &MmapMut) -> Result<(), DataLoaderError> {
        for &advice in &self.advice {
            mmap.advise(advice).map_err(|source| {
                // Map the io::Error to your custom DataLoaderError
                DataLoaderError::AdviseFailed { source, advice }
            })?;
        }
        Ok(())
    }
}

impl Default for AdviceSet {
    fn default() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self {
                advice: vec![memmap2::Advice::Sequential, memmap2::Advice::HugePage],
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Self::empty()
        }
    }
}
impl From<Vec<Advice>> for AdviceSet {
    fn from(v: Vec<Advice>) -> Self {
        Self { advice: v }
    }
}

impl From<&[Advice]> for AdviceSet {
    fn from(s: &[Advice]) -> Self {
        Self { advice: s.to_vec() }
    }
}

impl<const N: usize> From<[Advice; N]> for AdviceSet {
    fn from(a: [Advice; N]) -> Self {
        Self { advice: a.to_vec() }
    }
}

impl From<Advice> for AdviceSet {
    fn from(a: Advice) -> Self {
        Self { advice: vec![a] }
    }
}
