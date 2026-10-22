//! The integration tests, as modules of one test binary so that they link
//! once and run in parallel with each other.

mod checker_fuzz;
mod checker_library;
mod common;
mod finseq;
mod lighting;
mod precompute;
mod strip;
mod tracing;
