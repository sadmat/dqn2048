use rand::Rng;

use crate::game::board::NUM_TILES;

pub(crate) trait GameRng {
    fn new_tile_value(&self) -> u32;
    fn new_tile_position(&self) -> usize;
}

pub(crate) struct RealGameRng {}

impl RealGameRng {
    pub(crate) fn new() -> Self {
        RealGameRng {}
    }
}

impl GameRng for RealGameRng {
    fn new_tile_value(&self) -> u32 {
        if rand::rng().random::<f64>() <= 0.1 {
            4
        } else {
            2
        }
    }

    fn new_tile_position(&self) -> usize {
        rand::rng().random_range(0..NUM_TILES)
    }
}

pub(crate) struct FakeGameRng {
    tile_value: u32,
    tile_position: usize,
}

impl FakeGameRng {
    pub(crate) fn new(tile_value: u32, tile_position: usize) -> Self {
        FakeGameRng {
            tile_value,
            tile_position,
        }
    }
}

impl GameRng for FakeGameRng {
    fn new_tile_value(&self) -> u32 {
        self.tile_value
    }

    fn new_tile_position(&self) -> usize {
        self.tile_position
    }
}
