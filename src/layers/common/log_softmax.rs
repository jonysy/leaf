//! Computes the logarithmic softmax of its input.
//!

use layer::*;

use crate::typedefs::{ArcLockTensor, LeafBackend};
use parenchyma::prelude::SharedTensor;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// LogSoftmax Layer
pub struct LogSoftmax;

impl LayerWorker for LogSoftmax {
    fn reshape(&mut self,
               backend: ::std::rc::Rc<LeafBackend>,
               input_data: &mut Vec<ArcLockTensor>,
               input_gradient: &mut Vec<ArcLockTensor>,
               weights_data: &mut Vec<ArcLockTensor>,
               weights_gradient: &mut Vec<ArcLockTensor>,
               output_data: &mut Vec<ArcLockTensor>,
               output_gradient: &mut Vec<ArcLockTensor>) {
        let input_read = input_data[0].read().unwrap();
        let input_desc = input_read.shape();
        input_gradient[0].write().unwrap().resize(input_desc.clone()).unwrap();
        output_data[0].write().unwrap().resize(input_desc.clone()).unwrap();
        output_gradient[0].write().unwrap().resize(input_desc.clone()).unwrap();
    }
}

impl ComputeOutput<f32> for LogSoftmax {
    fn compute_output(&self,
                      backend: &LeafBackend,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        backend.log_softmax(input_data[0], output_data[0]).unwrap();
    }
}

impl ComputeInputGradient<f32> for LogSoftmax {
    fn compute_input_gradient(&self,
                              backend: &LeafBackend,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        backend.log_softmax_grad(output_data[0], output_gradients[0], input_gradients[0]).unwrap();
    }
}

impl ComputeParametersGradient<f32> for LogSoftmax { }

impl ::std::default::Default for LogSoftmax {
    fn default() -> LogSoftmax {
        LogSoftmax
    }
}
