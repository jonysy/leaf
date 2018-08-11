//! Flattens the bottom Blob into a simpler top Blob.
//!
//! Input of shape n * c * h * w becomes
//! a simple vector output of shape n * (c*h*w).
//!

/// Flattening Utility Layer
#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
pub struct Flatten;
