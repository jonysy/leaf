//! TODO: DOC
//!

use layer::*;
use leaf_capnp::negative_log_likelihood_config as capnp_config;
use capnp_util::*;

use crate::typedefs::{ArcLockTensor, LeafBackend};
use parenchyma::prelude::SharedTensor;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// NegativeLogLikelihood Loss Layer
pub struct NegativeLogLikelihood {
    num_classes: usize,
}

impl NegativeLogLikelihood {
    /// Create a NegativeLogLikelihood layer from a NegativeLogLikelihoodConfig.
    pub fn from_config(config: &NegativeLogLikelihoodConfig) -> NegativeLogLikelihood {
        NegativeLogLikelihood {
            num_classes: config.num_classes,
        }
    }

    fn calculate_outer_num(softmax_axis: usize, input_shape: &[usize]) -> usize {
        input_shape.iter().take(softmax_axis + 1).fold(1, |prod, i| prod * i)
    }

    fn calculate_inner_num(softmax_axis: usize, input_shape: &[usize]) -> usize {
        input_shape.iter().skip(softmax_axis + 1).fold(1, |prod, i| prod * i)
    }

    fn batch_size(input_shape: &[usize]) -> usize {
        match input_shape.len() {
            1 => 1,
            2 => input_shape[0],
            _ => panic!("NegativeLogLikelihood layer only supports 1D/2D inputs")
        }
    }
}

impl ILayer for NegativeLogLikelihood {
    impl_ilayer_loss!();

    fn sync_native(&self) -> bool {
        true
    }

    fn reshape(&mut self,
               backend: ::std::rc::Rc<LeafBackend>,
               input_data: &mut Vec<ArcLockTensor>,
               input_gradient: &mut Vec<ArcLockTensor>,
               weights_data: &mut Vec<ArcLockTensor>,
               weights_gradient: &mut Vec<ArcLockTensor>,
               output_data: &mut Vec<ArcLockTensor>,
               output_gradient: &mut Vec<ArcLockTensor>) {
        let data = input_data[0].read().unwrap();
        let label = input_data[1].read().unwrap();

        input_gradient[0].write().unwrap().resize(data.shape().clone()).unwrap();
        output_data[0].write().unwrap().resize(label.shape().clone()).unwrap();
    }
}

impl ComputeOutput<f32> for NegativeLogLikelihood {
    fn compute_output(&self,
                      backend: &LeafBackend,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let probabilities = input_data[0];
        let labels = input_data[1];

        let batch_size = Self::batch_size(labels.shape().dimensions());

        let native_labels = labels.as_slice().unwrap();
        let native_probabilities = probabilities.as_slice().unwrap();

        let mut writable_loss = Vec::<f32>::new();
        for &label_value in native_labels {
            let probability_value = native_probabilities[label_value as usize];
            writable_loss.push(-probability_value);
        }

        let mut loss = writable_loss.iter().fold(0f32, |sum, &val| sum + val);
        loss = loss / (batch_size as f32);
        writable_loss = vec![loss];

        output_data[0].write_slice(&writable_loss[..]).unwrap()
    }
}

impl ComputeInputGradient<f32> for NegativeLogLikelihood {
    fn compute_input_gradient(&self,
                              backend: &LeafBackend,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let labels = input_data[1];
        let batch_size = Self::batch_size(input_data[0].shape().dimensions());
        let num_classes = self.num_classes;

        let native_labels = labels.as_slice().unwrap();
        let mut writable_gradient = vec![0f32; input_gradients[0].shape().capacity()];

        for (batch_n, &label_value) in native_labels.iter().enumerate() {
            let index = (num_classes * batch_n) + label_value as usize;
            writable_gradient[index] = -1f32;
        }
        input_gradients[0].write_slice(&writable_gradient[..]).unwrap()
    }
}

impl ComputeParametersGradient<f32> for NegativeLogLikelihood { }

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Specifies configuration parameters for a NegativeLogLikelihood Layer.
pub struct NegativeLogLikelihoodConfig {
    /// How many different classes can be classified.
    pub num_classes: usize,
}

impl<'a> CapnpWrite<'a> for NegativeLogLikelihoodConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the NegativeLogLikelihoodConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_num_classes(self.num_classes as u64);
    }
}

impl<'a> CapnpRead<'a> for NegativeLogLikelihoodConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let num_classes = reader.get_num_classes() as usize;

        NegativeLogLikelihoodConfig {
            num_classes: num_classes
        }
    }
}

impl Into<LayerType> for NegativeLogLikelihoodConfig {
    fn into(self) -> LayerType {
        LayerType::NegativeLogLikelihood(self)
    }
}
