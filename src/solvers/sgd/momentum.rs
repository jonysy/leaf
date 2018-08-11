//! A [Stochastic Gradient Descent with Momentum][1]
//! [1]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
//!
//! Momentum in solving neural networks works similar to
//! they way it does in physics.
//! If you travel into a a direction with a high velocity,
//! it becomes very hard to change (or reverse)
//! the direction in which you are moving.
//!
//! Similarly when adjusting gradients during solving,
//! keeping a part of the previous gradient update can make solving faster,
//! since if you keep adjusting the gradients
//! into the same direction you will reach the optimum faster.
//! It also makes solving more stable.

use layer::*;
use solver::*;
use solvers::SGDSolver;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::typedefs::{ArcLockTensor, LeafBackend};
use parenchyma::prelude::SharedTensor;

#[derive(Debug)]
/// Stochastic Gradient Descent with Momentum.
///
/// See [module description][1] for more information.
/// [1]: ./index.html
pub struct Momentum {
    /// The gradient update from the previous iteration for each blob.
    history: Vec<ArcLockTensor>,
    /// The backend used for computing the gradient.
    backend: Rc<LeafBackend>,

    /// Scalar that temporarily holds learing rate for weight update computations
    lr: SharedTensor<f32>,
    /// Scalar that temporarily holds momentum for weight update computations
    momentum: SharedTensor<f32>,
}

impl Momentum {
    /// Create a new SGD Momentum solver.
    ///
    /// Should not be called directly.
    /// Use [Solver::from_config][2] instead.
    ///
    /// [2]: ../../../solver/struct.Solver.html#method.from_config
    pub fn new(backend: Rc<LeafBackend>) -> Momentum {
        let (lr, momentum) = {
            (SharedTensor::<f32>::from(1),
             SharedTensor::<f32>::from(1))
        };
        
        Momentum {
            history: Vec::new(),
            backend: backend,

            lr: lr,
            momentum: momentum,
        }
    }

}

impl SGDSolver for Momentum {
    fn compute_update_value(&mut self,
                            config: &SolverConfig,
                            weight_gradient: &ArcLockTensor,
                            history_blob_id: usize,
                            global_lr: &f32,
                            blob_lr: &f32) {
        ::weight::FillerType::Constant {
            value: global_lr * blob_lr
        }.fill(&mut self.lr);

        ::weight::FillerType::Constant {
            value: config.momentum
        }.fill(&mut self.momentum);

        let backend = SolverWorker::backend(self);

        let history_blob = &self.history[history_blob_id];

        backend.axpby(
            &self.lr,
            &weight_gradient.read().unwrap(),
            &self.momentum,
            &mut history_blob.write().unwrap()
        ).unwrap();

        backend.copy(
            &history_blob.read().unwrap(), 
            &mut weight_gradient.write().unwrap()
        ).unwrap();
    }
}

impl_isolver_sgd!(Momentum);
