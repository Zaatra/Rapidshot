//! Capture, conversion and transfer for RapidShot, with no Python in it.
//!
//! `rapidshot-native` (the Python extension) is a thin binding over this crate,
//! and the Rust remote-support helper links it directly, so neither consumer
//! repeats the other's logic. Anything that needs the interpreter -- argument
//! parsing, `dict`s, releasing the GIL -- stays in the binding.

pub mod capture_order;
pub mod converter12;
pub mod cross_adapter;
pub mod desktop;
pub mod duplicator;
pub mod luma;
pub mod preprocess;
pub mod preprocess12;
pub mod report;
pub mod swizzle;
pub mod video_processor;
pub mod wgc;
