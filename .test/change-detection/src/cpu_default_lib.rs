#[allow(dead_code)]
mod consumer;
mod cpu_default_compare;
mod cpu_default_replay;
#[allow(dead_code)]
mod delivery;
#[allow(dead_code)]
mod fingerprint;
#[allow(dead_code)]
mod gpu;

use gpu::{Gpu, GpuFrame};

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

const TILE_SIZE: u32 = 128;

#[derive(Clone, Copy)]
struct Extent {
    width: u32,
    height: u32,
}

impl Extent {
    fn tile_columns(self) -> u32 {
        self.width.div_ceil(TILE_SIZE)
    }

    fn tile_count(self) -> u32 {
        self.tile_columns() * self.height.div_ceil(TILE_SIZE)
    }
}
