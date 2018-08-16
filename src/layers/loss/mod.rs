//! Provides methods to calculate the loss (cost) of some output.
//!
//! A loss function is also sometimes called cost function.

pub use self::negative_log_likelihood::{NegativeLogLikelihood, NegativeLogLikelihoodConfig};

pub mod negative_log_likelihood;
