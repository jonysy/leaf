//! Applies the nonlinear Rectified Linear Unit.
//!
//! Non-linearity activation function: y = max(0, x)
//!
//! This is generally the preferred choice over Sigmoid or TanH.
//! The max function used in ReLU is usually faster to compute than the exponentiation
//! needed in a Sigmoid layer.

use crate::layers::core::{ComputeInputGradient, ComputeOutput, ComputeParametersGradient};
use crate::typedefs::LeafBackend;

use parenchyma::prelude::SharedTensor;

/// ReLU Activation Layer
#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
pub struct ReLU;

impl super::ActivationLayer for ReLU {
    // ..
}

impl ComputeOutput<f32> for ReLU {
    fn compute_output(&self,
        backend: &LeafBackend,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>]) {

        match input_data.get(0) {
            Some(input) => backend.relu(input, output_data[0]).unwrap(),

            None => {
                panic!("No input provided for ReLU layer.")
                // TODO
                // backend.relu_pointwise(output_data[0]).unwrap(),
            }
        }
    }
}

impl ComputeInputGradient<f32> for ReLU {
    fn compute_input_gradient(&self,
        backend: &LeafBackend,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>]) {

        match output_data.get(0) {
            Some(_) => {
                backend.relu_grad(
                    output_data[0], 
                    output_gradients[0], 
                    input_data[0], 
                    input_gradients[0]
                ).unwrap()
            }
            None => {
                panic!("No output_data provided for ReLU layer backward.")
                // TODO
                // backend.relu_pointwise_grad(input_data[0], input_gradients[0]).unwrap(),
            }
        }
    }
}

impl ComputeParametersGradient<f32> for ReLU {
    // ..
}