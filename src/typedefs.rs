//! Convenient typedefs

use parenchyma::prelude::{Backend, SharedTensor};
use parenchyma_ml::Package as MachLrnPackage;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A shared lock used for tensors.
///
/// This typedef is used to avoid writing out `Arc<RwLock<SharedTensor<T>>>`.
pub type ArcLockTensor<T = f32> = Arc<RwLock<SharedTensor<T>>>;

/// Leaf stores, processes, and passes along data using blobs. 
/// (todo? also provides synchronization capability between the CPU and the GPU)
///
/// A blob is a N-D data structure that contains data, gradients, or weights (including biases).
///
/// todo `blob` wasn't descriptive enough.. `envelope` won't cut it either.
pub type ArcLockTensorBlob<T = f32> = (ArcLockTensor<T>, ArcLockTensor<T>);

/// A backend extended with the Parenchyma BLAS extension package.
///
/// The ML package bundles the BLAS and NN packages together to provide the backend with 
/// both BLAS and NN operations.
pub type LeafBackend = Backend<MachLrnPackage>;

/// A shared lock used for tensors.
pub type Registry<V> = HashMap<String, V>;

/// Represents a weight blob.
pub type WeightArcLockTensorBlob<T = f32> = (ArcLockTensor<T>, ArcLockTensor<T>, Option<T>, Option<T>);