//! Data type conversion traits for the dataloader.
//!
//! This module defines how raw byte chunks retrieved from the storage layer
//! are transformed into the specific data types required by the application,
//! such as framework-specific tensor representations or raw byte buffers.

use bytes::Bytes;

/// A trait for converting raw bytes into a specific output type.
///
/// Types implementing this trait can be used with the [`Dataloader`](crate::dataloader::loader::Dataloader) to
/// automatically transform data as it is read from the underlying storage.
///
/// # Examples
///
/// Implementing a custom converter for `u32` arrays:
///
/// ```rust,ignore
/// use bytes::Bytes;
/// use plast::dataloader::datatypes::DataloaderType;
///
/// struct U32Converter;
///
/// impl DataloaderType for U32Converter {
///     type Output = Vec<u32>;
///
///     fn convert(bytes: Bytes) -> Self::Output {
///         bytes.chunks_exact(4).map(|chunk| {
///             u32::from_le_bytes(chunk.try_into().unwrap())
///         }).collect()
///     }
/// }
/// ```
pub trait DataloaderType {
    /// The resulting type after conversion.
    type Output;

    /// Converts the raw `Bytes` into the associated `Output` type.
    fn convert(bytes: Bytes) -> Self::Output;
}

/// A converter that transforms raw bytes into Burn's specific `Bytes` type.
///
/// This is useful when using the `burn` framework, allowing the dataloader
/// to feed directly into Burn tensors without manual conversion overhead.
///
/// Requires the `burn` feature to be enabled.
#[cfg(feature = "burn")]
#[derive(Debug, Clone, Default)]
pub struct BurnBytesConverter;

#[cfg(feature = "burn")]
impl DataloaderType for BurnBytesConverter {
    type Output = burn_tensor::Bytes;

    #[inline]
    fn convert(bytes: Bytes) -> Self::Output {
        burn_tensor::Bytes::from_shared(bytes, burn_tensor::AllocationProperty::File)
    }
}

/// A simple pass-through converter that returns the raw bytes unmodified.
///
/// Use this when you want to handle the byte deserialization or tensor
/// conversion manually after receiving the data from the dataloader.
///
/// # Examples
///
/// ```rust,ignore
/// use bytes::Bytes;
/// use plast::dataloader::datatypes::{BytesConverter, DataloaderType};
///
/// let raw_bytes = Bytes::from(vec![1, 2, 3, 4]);
/// let output = BytesConverter::convert(raw_bytes.clone());
/// assert_eq!(output, raw_bytes);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BytesConverter;

impl DataloaderType for BytesConverter {
    type Output = Bytes;

    #[inline(always)]
    fn convert(bytes: Bytes) -> Self::Output {
        bytes
    }
}
