//! Applies the nonlinear Log-Sigmoid function.
//!
//! Non-linearity activation function: y = (1 + e^(-x))^(-1)
//!
//! A classic choice in neural networks.
//! But you might consider using ReLu as an alternative.
//!
//! ReLu, compared to Sigmoid
//!
//! * reduces the likelyhood of vanishing gradients
//! * increases the likelyhood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this
//! writing (2015).

use layer::*;

use crate::typedefs::{ArcLockTensor, LeafBackend};
use parenchyma::prelude::SharedTensor;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Sigmoid Activation Layer
pub struct Sigmoid;

//
// Sigmoid + SigmoidPointwise
// Only on CUDA
//
// Sigmoid without SigmoidPointwise
// Only on Native
//
impl LayerWorker for Sigmoid {
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

impl ComputeOutput<f32> for Sigmoid {
    fn compute_output(&self,
        backend: &LeafBackend,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>]) {

        match input_data.get(0) {
            Some(input) => backend.sigmoid(input, output_data[0]).unwrap(),
            None => {
                panic!("No input provided for Sigmoid layer.")
                // TODO
                // backend.sigmoid_pointwise(output_data[0]).unwrap()
            }
        }
    }
}

impl ComputeInputGradient<f32> for Sigmoid {
    fn compute_input_gradient(&self,
        backend: &LeafBackend,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>]) {

        match output_data.get(0) {
            Some(_) => backend.sigmoid_grad(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => {
                panic!("No output_data provided for Sigmoid layer backward.")
                // TODO
                // backend.sigmoid_pointwise_grad(input_data[0], input_gradients[0]).unwrap()
            }
        }
    }
}

impl ComputeParametersGradient<f32> for Sigmoid {}
