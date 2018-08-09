//! A collection of helper/utility functions.

use num::traits::{NumCast, cast};
use parenchyma::prelude::SharedTensor;

/// Writes the `i`th sample of a batch into a `SharedTensor`.
///
/// The size of a single sample is inferred through the first dimension of the SharedTensor, which
/// is assumed to be the size of the batch.
///
/// Allocates memory on a Native Backend if necessary.
pub fn write_batch_sample<T>(tensor: &mut SharedTensor, data: &[T], i: usize)
    where T: Copy + NumCast {
    let batch_size = tensor.shape().capacity();
    let sample_size = batch_size / tensor.shape().dimensions()[0];
    tensor.write_offset_iter(data.iter().map(|elem| cast(*elem).unwrap()), i * sample_size).unwrap()
}