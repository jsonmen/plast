use bytes::Bytes;

pub trait DataloaderType {
    type Output;
    fn convert(bytes: Bytes) -> Self::Output;
}

#[cfg(feature = "burn")]
#[derive(Debug, Clone, Default)]
pub struct BurnBytesConverter;

#[cfg(feature = "burn")]
impl DataloaderType for BurnBytesConverter {
    type Output = burn_tensor::Bytes;

    fn convert(bytes: Bytes) -> Self::Output {
        burn_tensor::Bytes::from_shared(bytes, burn_tensor::AllocationProperty::File)
    }
}
#[derive(Debug, Clone, Default)]
pub struct BytesConverter;

impl DataloaderType for BytesConverter {
    type Output = Bytes;

    fn convert(bytes: Bytes) -> Self::Output {
        bytes
    }
}
