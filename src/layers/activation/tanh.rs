//! Applies the nonlinear TanH function.
//!
//! Non-linearity activation function: y = sinh(x) / cosh(x)
//!
//! You might consider using ReLU as an alternative.
//!
//! ReLU, compared to TanH:
//!
//! * reduces the likelihood of vanishing gradients
//! * increases the likelihood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this writing (2016).

use crate::layers::core::{ComputeInputGradient, ComputeOutput, ComputeParametersGradient};
use crate::typedefs::LeafBackend;

use parenchyma::prelude::SharedTensor;

/// TanH Activation Layer
#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
pub struct TanH;

impl super::ActivationLayer for TanH {
    // ..
}

impl ComputeOutput<f32> for TanH {
    fn compute_output(&self,
        backend: &LeafBackend,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>]) {

        match input_data.get(0) {
            Some(input) => backend.tanh(input, output_data[0]).unwrap(),
            
            None => {
              panic!("No input provided for TanH layer.")
              // TODO
              // backend.tanh_pointwise(output_data[0]).unwrap()
            }
        }
    }
}

impl ComputeInputGradient<f32> for TanH {
    fn compute_input_gradient(&self,
        backend: &LeafBackend,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>]) {
        
        match output_data.get(0) {
            Some(_) => {
                backend.tanh_grad(
                    output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap()
            }

            None => {
              panic!("No output_data provided for TanH layer backward.")
              // TODO
              // backend.tanh_pointwise_grad(input_data[0], input_gradients[0]).unwrap()
            }
        }
    }
}

impl ComputeParametersGradient<f32> for TanH {
    // ..
}
