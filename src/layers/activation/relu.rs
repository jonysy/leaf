//! Applies the nonlinear Rectified Linear Unit.
//!
//! Non-linearity activation function: y = max(0, x)
//!
//! This is generally the preferred choice over Sigmod or TanH.
//! The max function used in ReLU is usually faster to compute than the exponentiation
//! needed in a Sigmoid layer.

use layer::*;

use crate::typedefs::{ArcLockTensor, LeafBackend};
use parenchyma::prelude::SharedTensor;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// ReLU Activation Layer
pub struct ReLU;

//
// ReLU + ReLUPointwise
// Only on CUDA
//
// ReLU without ReLUPointwise
// Only on Native
//
impl LayerWorker for ReLU {
    impl_ilayer_activation!();

    fn reshape(&mut self,
        backend: ::std::rc::Rc<LeafBackend>,
        input_data: &mut Vec<ArcLockTensor>,
        input_gradient: &mut Vec<ArcLockTensor>,
        weights_data: &mut Vec<ArcLockTensor>,
        weights_gradient: &mut Vec<ArcLockTensor>,
        output_data: &mut Vec<ArcLockTensor>,
        output_gradient: &mut Vec<ArcLockTensor>) {

        if let Some(inp) = input_data.get(0) {
            let read_inp = inp.read().unwrap();
            let input_desc = read_inp.shape();
            input_gradient[0].write().unwrap().resize(input_desc.clone()).unwrap();
            output_data[0].write().unwrap().resize(input_desc.clone()).unwrap();
            output_gradient[0].write().unwrap().resize(input_desc.clone()).unwrap();
        }
    }
}

impl ComputeOutput<f32> for ReLU {
    fn compute_output(&self,
        backend: &LeafBackend,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>]) {

        match input_data.get(0) {
            Some(input) => {
                backend.relu(input, output_data[0]).unwrap()
            }
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

impl ComputeParametersGradient<f32> for ReLU {}