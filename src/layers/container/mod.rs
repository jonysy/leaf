//! Provides container layers.
//!
//! For now layers in container should be described as layers that are used
//! to connect multiple layers together to create 'networks'.

pub use self::sequential::{Sequential, SequentialConfig};

pub mod sequential;
