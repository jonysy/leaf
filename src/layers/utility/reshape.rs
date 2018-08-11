//! Utility layer to give a tensor another shape.
//!
//! This layer should be used as in-place operation,
//! so the tensor that should be reshaped should be specified
//! as both input and output.
//!
//! Reshaping a tensor is required so that it becomes
//! usable for Layers that interpret meaning into the shape of
//! the tensor.
//!
//! A lot of layers interpret the last dimensions as NCHW,
//! where the letters stand for:
//!
//! - `N` : number of batch samples
//! - `C` : number of feature maps
//! - `H` : height
//! - `W` : width

use crate::cerealization_protocol::*;
use crate::cerealization_protocol::reshape_config as capnp_config;
use crate::layers::core::*;
use crate::typedefs::{ArcLockTensor, LeafBackend};

use parenchyma::prelude::SharedTensor;

/// Reshape Utility Layer
#[derive(Clone, Debug)]
pub struct Reshape{
    shape: Vec<usize>,
}

impl Reshape {
    /// Create a Reshape layer from a ReshapeConfig.
    pub fn from_config(config: &ReshapeConfig) -> Reshape {
        Reshape {
            shape: config.shape.clone(),
        }
    }
}

impl LayerWorker for Reshape {
    fn compute_in_place(&self) -> bool {
        true
    }

    fn auto_output_blobs(&self) -> bool {
        false
    }

    fn reshape(&mut self,
               backend: ::std::rc::Rc<LeafBackend>,
               input_data: &mut Vec<ArcLockTensor>,
               input_gradient: &mut Vec<ArcLockTensor>,
               weights_data: &mut Vec<ArcLockTensor>,
               weights_gradient: &mut Vec<ArcLockTensor>,
               output_data: &mut Vec<ArcLockTensor>,
               output_gradient: &mut Vec<ArcLockTensor>) {
        output_data[0].write().unwrap().resize(&self.shape[..]).unwrap();
        output_gradient[0].write().unwrap().resize(&self.shape[..]).unwrap();
    }
}

impl ComputeOutput<f32> for Reshape {
    fn compute_output(&self,
                      backend: &LeafBackend,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
    }
}

impl ComputeInputGradient<f32> for Reshape {
    fn compute_input_gradient(&self,
                              backend: &LeafBackend,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {}
}

impl ComputeParametersGradient<f32> for Reshape {}

#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Reshape Layer.
pub struct ReshapeConfig {
    /// The target shape that the input should assume.
    ///
    /// Preceding dimensions are treated as independent inputs
    ///
    /// Defaults to `1`
    pub shape: Vec<usize>,
}

impl ReshapeConfig {
    /// Create a ReshapeConfig that describes a Reshape layer with a provided shape.
    pub fn of_shape(shape: &[usize]) -> ReshapeConfig {
        ReshapeConfig {
            shape: shape.to_owned()
        }
    }
}

impl<'a> CapnpWrite<'a> for ReshapeConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the ReshapeConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        let mut shape = builder.borrow().init_shape(self.shape.len() as u32);
        for (i, dim) in self.shape.iter().enumerate() {
            shape.set(i as u32, *dim as u64);
        }
    }
}

impl<'a> CapnpRead<'a> for ReshapeConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let read_shape = reader.get_shape().unwrap();
        let mut shape = Vec::new();
        for i in 0..read_shape.len() {
            shape.push(read_shape.get(i) as usize)
        }

        ReshapeConfig {
            shape: shape
        }
    }
}

impl Into<LayerType> for ReshapeConfig {
    fn into(self) -> LayerType {
        LayerType::Reshape(self)
    }
}
