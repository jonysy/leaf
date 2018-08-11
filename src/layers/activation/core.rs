use crate::layers::core::{ComputeInputGradient, ComputeOutput, ComputeParametersGradient, LayerWorker};
use crate::typedefs::{ArcLockTensor, LeafBackend};

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