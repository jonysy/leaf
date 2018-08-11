//! Applies the nonlinear TanH function.
//!
//! Non-linearity activation function: y = sinh(x) / cosh(x)
//!
//! You might consider using ReLU as an alternative.
//!
//! ReLU, compared to TanH:
//!
//! * reduces the likelyhood of vanishing gradients
//! * increases the likelyhood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this writing (2016).

use layer::*;

use crate::typedefs::{ArcLockTensor, LeafBackend};
use parenchyma::prelude::SharedTensor;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// TanH Activation Layer
pub struct TanH;

//
// Tanh + TanhPointwise
// Only on CUDA
//
// Tanh without TanhPointwise
// Only on Native
//
impl ILayer for TanH {
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
            Some(_) => backend.tanh_grad(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap(),
            None => {
              panic!("No output_data provided for TanH layer backward.")
              // TODO
              // backend.tanh_pointwise_grad(input_data[0], input_gradients[0]).unwrap()
            }
        }
    }
}

impl ComputeParametersGradient<f32> for TanH {}
