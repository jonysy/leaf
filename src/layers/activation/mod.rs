//! Provides nonlinear activation methods.
//!
//! Activation Layers take a input tensor, provide the activation operation and
//! produce a output tensor.
//! Thanks to the nonlinearity of the activation methods, we can 'learn' and
//! detect nonlinearities
//! in our (complex) datasets.
//!
//! The activation operation used should depend on the task at hand. For binary
//! classification a
//! step function might be very useful. For more complex tasks continious
//! activation functions such
//! as [Sigmoid][mod_sigmoid], TanH, [ReLU][mod_relu] should be used. In most cases ReLU might
//! provide the best results.
//!
//! If you supply the same blob as input and output to a layer via the [LayerConfig][struct_layerconfig],
//! computations will be done in-place, requiring less memory.
//!
//! The activation function is also sometimes called transfer function.
//!
//! [mod_sigmoid]: ./sigmoid/index.html
//! [mod_relu]: ./relu/index.html
//! [struct_layerconfig]: ../../layer/struct.LayerConfig.html

pub use self::relu::ReLU;
pub use self::sigmoid::Sigmoid;
pub use self::tanh::TanH;

pub mod relu;
pub mod sigmoid;
pub mod tanh;

use crate::typedefs::{ArcLockTensor, LeafBackend};
use super::core::{ComputeInputGradient, ComputeOutput, ComputeParametersGradient, LayerWorker};

/// A reusable "partial" trait (see default `LayerWorker` functions) implemented by all activation 
/// layers and provides default functions.
///
/// <Activation function> + ReLUPointwise (Only on CUDA)
/// <Activation function> without ReLUPointwise (Only on Native)
pub trait ActivationLayer: ComputeParametersGradient<f32> 
    + ComputeOutput<f32>
    + ComputeInputGradient<f32>  {
    // ...
}

impl<A> LayerWorker for A where A: ActivationLayer {
    fn exact_num_output_blobs(&self) -> Option<usize> {
        return Some(1);
    }

    fn exact_num_input_blobs(&self) -> Option<usize> {
        return Some(1);
    }

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