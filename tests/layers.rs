extern crate leaf;

#[cfg(test)]
mod layers_spec {
    use leaf::layers::{ReLU, Sigmoid, TanH};
    use leaf::layer::LayerWorker;

    #[test]
    fn test_exact_num_input_and_output_blobs_for_a_relu_layer() {
        assert_eq!(ReLU.exact_num_output_blobs(), Some(1));
        assert_eq!(ReLU.exact_num_input_blobs(), Some(1));
    }

    #[test]
    fn test_exact_num_input_and_output_blobs_for_a_sigmoid_layer() {
        assert_eq!(Sigmoid.exact_num_output_blobs(), Some(1));
        assert_eq!(Sigmoid.exact_num_input_blobs(), Some(1));
    }

    #[test]
    fn test_exact_num_input_and_output_blobs_for_a_tanh_layer() {
        assert_eq!(TanH.exact_num_output_blobs(), Some(1));
        assert_eq!(TanH.exact_num_input_blobs(), Some(1));
    }
}