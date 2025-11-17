use rand::prelude::*;

use crate::game::game_rng::{GameRng, RealGameRng};

#[derive(Copy, Clone, PartialEq, Debug)]
pub(crate) enum Tile {
    Empty,
    Value(u32),
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub(crate) enum Direction {
    Up,
    Down,
    Left,
    Right,
}

pub(crate) const NUM_ROWS: usize = 4;
pub(crate) const NUM_COLUMNS: usize = 4;
pub(crate) const NUM_TILES: usize = NUM_ROWS * NUM_COLUMNS;

#[derive(Clone)]
pub(crate) struct Board<Rng: GameRng> {
    pub(crate) score: u32,
    tiles: [Tile; NUM_TILES],
    rng: Rng,
}

impl Board<RealGameRng> {
    pub fn new() -> Board<RealGameRng> {
        let mut board = Board {
            score: 0,
            tiles: [Tile::Empty; NUM_TILES],
            rng: RealGameRng::new(),
        };
        board.place_random_tile();
        board.place_random_tile();
        board
    }
}

impl<Rng: GameRng> Board<Rng> {
    fn new_with_tiles(tiles: [Tile; NUM_TILES], rng: Rng) -> Board<Rng> {
        Board {
            score: 0,
            tiles: tiles,
            rng: rng,
        }
    }

    fn place_random_tile(&mut self) {
        // TODO: check for full board
        loop {
            let index = self.rng.new_tile_position();
            if self.tiles[index] == Tile::Empty {
                self.tiles[index] = Tile::Value(self.rng.new_tile_value());
                break;
            }
        }
    }

    pub fn move_right(&mut self) {
        if !self.can_move_right() {
            return;
        }
        let mut merged_tiles = [false; NUM_TILES];
        for column in (0..NUM_COLUMNS - 1).rev() {
            for row in 0..NUM_ROWS {
                self.slide_tile(row, column, Direction::Right, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    pub fn move_left(&mut self) {
        if !self.can_move_left() {
            return;
        }
        let mut merged_tiles = [false; NUM_TILES];
        for column in 1..NUM_COLUMNS {
            for row in 0..NUM_ROWS {
                self.slide_tile(row, column, Direction::Left, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    pub fn move_down(&mut self) {
        if !self.can_move_down() {
            return;
        }
        let mut merged_tiles = [false; NUM_TILES];
        for row in (0..NUM_ROWS - 1).rev() {
            for column in 0..NUM_COLUMNS {
                self.slide_tile(row, column, Direction::Down, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    pub fn move_up(&mut self) {
        if !self.can_move_up() {
            return;
        }
        let mut merged_tiles = [false; NUM_TILES];
        for row in 1..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                self.slide_tile(row, column, Direction::Up, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    fn slide_tile(
        &mut self,
        row: usize,
        column: usize,
        direction: Direction,
        merged_tiles: &mut [bool; NUM_TILES],
    ) {
        let source_index = index(row, column);
        let Tile::Value(source_tile) = self.tiles[source_index] else {
            return;
        };

        let (dx, dy) = direction.vector();
        let mut target_row = row;
        let mut target_column = column;

        // while (0..NUM_ROWS).contains(&(target_row + ))
        //     && (0..NUM_COLUMNS).contains(&(target_column + 1))
        loop {
            let next_row = (target_row as i32 + dy as i32) as usize;
            let next_column = (target_column as i32 + dx as i32) as usize;

            if !(0..NUM_ROWS).contains(&next_row) || !(0..NUM_COLUMNS).contains(&next_column) {
                break;
            }

            let next_index = index(next_row, next_column);

            match self.tiles[next_index] {
                Tile::Empty => {
                    target_row = next_row;
                    target_column = next_column;
                }
                Tile::Value(target_value) => {
                    if target_value == source_tile && !merged_tiles[next_index] {
                        target_row = next_row;
                        target_column = next_column;
                    }
                    break;
                }
            }
        }

        let target_index = index(target_row, target_column);
        if target_index == source_index {
            return;
        } else if let Tile::Value(target_value) = self.tiles[target_index] {
            self.tiles[target_index] = Tile::Value(source_tile + target_value);
            self.tiles[source_index] = Tile::Empty;
            merged_tiles[target_index] = true;
            self.score += source_tile + target_value;
        } else {
            self.tiles[target_index] = self.tiles[source_index];
            self.tiles[source_index] = Tile::Empty;
        }
    }

    pub fn is_over(&self) -> bool {
        let empty_tiles_exist = self
            .tiles
            .iter()
            .filter(|tile| *tile == &Tile::Empty)
            .count()
            > 0;
        if empty_tiles_exist {
            return false;
        }

        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                let Tile::Value(tile_value) = self.tiles[index(row, column)] else {
                    continue;
                };

                if let Some(right_value) = self.value_at(row, column + 1)
                    && right_value == tile_value
                {
                    return false;
                } else if let Some(bottom_value) = self.value_at(row + 1, column)
                    && bottom_value == tile_value
                {
                    return false;
                }
            }
        }

        true
    }

    pub fn value_at(&self, row: usize, column: usize) -> Option<u32> {
        if !(0..NUM_ROWS).contains(&row) || !(0..NUM_COLUMNS).contains(&column) {
            return None;
        }

        match self.tiles[index(row, column)] {
            Tile::Empty => None,
            Tile::Value(value) => Some(value),
        }
    }

    pub fn can_move_up(&self) -> bool {
        for row in 1..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                if self.can_move_tile_at(row, column, Direction::Up) {
                    return true;
                }
            }
        }
        false
    }

    pub fn can_move_down(&self) -> bool {
        for row in 0..NUM_ROWS - 1 {
            for column in 0..NUM_COLUMNS {
                if self.can_move_tile_at(row, column, Direction::Down) {
                    return true;
                }
            }
        }
        false
    }

    pub fn can_move_left(&self) -> bool {
        for row in 0..NUM_ROWS {
            for column in 1..NUM_COLUMNS {
                if self.can_move_tile_at(row, column, Direction::Left) {
                    return true;
                }
            }
        }
        false
    }

    pub fn can_move_right(&self) -> bool {
        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS - 1 {
                if self.can_move_tile_at(row, column, Direction::Right) {
                    return true;
                }
            }
        }
        false
    }

    fn can_move_tile_at(&self, row: usize, column: usize, direction: Direction) -> bool {
        let Some(tile_value) = self.value_at(row, column) else {
            return false;
        };

        let (dx, dy) = direction.vector();
        let next_tile_row = (row as isize + dy) as usize;
        let next_tile_column = (column as isize + dx) as usize;

        match self.value_at(next_tile_row, next_tile_column) {
            Some(next_tile_value) => next_tile_value == tile_value,
            None => {
                (0..NUM_ROWS).contains(&next_tile_row)
                    && (0..NUM_COLUMNS).contains(&next_tile_column)
            }
        }
    }
}

fn index(row: usize, column: usize) -> usize {
    row * NUM_ROWS + column
}

impl Direction {
    fn vector(&self) -> (isize, isize) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::Zip;

    use crate::game::game_rng::FakeGameRng;

    use super::Tile::{Empty, Value};
    use super::*;

    struct MergeConfiguration {
        initial_tiles: [Tile; NUM_TILES],
        direction: Direction,
        expected_tiles: [Tile; NUM_TILES],
        expected_score: u32,
        new_tile_index: usize,
    }

    #[test]
    fn new_board_with_two_tiles() {
        let board = Board::new();

        assert_eq!(board.score, 0);
        assert_eq!(
            board
                .tiles
                .iter()
                .filter(|tile| {
                    match tile {
                        Tile::Empty => false,
                        Tile::Value(value) if *value == 2 || *value == 4 => true,
                        Tile::Value(value) => {
                            panic!("{} is not a correct tile value", value)
                        }
                    }
                })
                .count(),
            2
        );
    }

    #[test]
    fn move_right() {
        let new_tile_value = 2;
        let new_tile_index = 0;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Value(2), Empty,    Empty,    Empty,
            Value(2), Value(4), Empty,    Empty,
            Value(2), Value(4), Value(8), Empty,
            Value(2), Value(4), Value(8), Value(16),
        ], FakeGameRng::new(new_tile_value, new_tile_index));

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Value(2), Empty,    Empty,    Value(2), 
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Value(2), Value(4), Value(8),
            Value(2), Value(4), Value(8), Value(16),
        ];

        board.move_right();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn move_left() {
        let new_tile_value = 2;
        let new_tile_index = 3;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Empty,    Empty,    Empty,    Value(2), 
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Value(2), Value(4), Value(8),
            Value(2), Value(4), Value(8), Value(16),
        ], FakeGameRng::new(new_tile_value, new_tile_index));

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Value(2), Empty,    Empty,    Value(2),
            Value(2), Value(4), Empty,    Empty,
            Value(2), Value(4), Value(8), Empty,
            Value(2), Value(4), Value(8), Value(16),
        ];

        board.move_left();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn move_down() {
        let new_tile_value = 2;
        let new_tile_index = 0;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Value(2), Value(4), Value(8), Value(16),
            Empty,    Value(2), Value(4), Value(8),
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Empty,    Empty,    Value(2), 
        ], FakeGameRng::new(new_tile_value, new_tile_index));

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Value(2), Empty,    Empty,     Value(16),
            Empty,    Empty,    Value(8),  Value(8),
            Empty,    Value(4), Value(4),  Value(4), 
            Value(2), Value(2), Value(2),  Value(2), 
        ];

        board.move_down();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn move_up() {
        let new_tile_value = 2;
        let new_tile_index = 12;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Empty,    Empty,    Empty,     Value(16),
            Empty,    Empty,    Value(8),  Value(8),
            Empty,    Value(4), Value(4),  Value(4), 
            Value(2), Value(2), Value(2),  Value(2), 
        ], FakeGameRng::new(new_tile_value, new_tile_index));

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Value(2), Value(4), Value(8), Value(16),
            Empty,    Value(2), Value(4), Value(8),
            Empty,    Empty,    Value(2), Value(4), 
            Value(2), Empty,    Empty,    Value(2), 
        ];

        board.move_up();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn merging_tiles() {
        #[rustfmt::skip]
        let configurations: [MergeConfiguration; _] = [
            // Merge up
            MergeConfiguration {
                initial_tiles: [
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Empty,    Empty,    Empty,    Empty,
                    Empty,    Empty,    Empty,    Empty,
                ],
                direction: Direction::Up,
                expected_tiles: [
                    Value(4), Value(8), Value(16), Value(32),
                    Empty,    Empty,    Empty,    Empty,
                    Empty,    Empty,    Empty,    Empty,
                    Empty,    Empty,    Empty,    Value(2),
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 15,
            },
            // Merge down
            MergeConfiguration {
                initial_tiles: [
                    Empty,    Empty,    Empty,    Empty,
                    Empty,    Empty,    Empty,    Empty,
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                ],
                direction: Direction::Down,
                expected_tiles: [
                    Empty,    Empty,    Empty,    Value(2),
                    Empty,    Empty,    Empty,    Empty,
                    Empty,    Empty,    Empty,    Empty,
                    Value(4), Value(8), Value(16), Value(32),
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 3,
            },
            // Merge left
            MergeConfiguration {
                initial_tiles: [
                    Value(2),  Value(2),  Empty, Empty,
                    Value(4),  Value(4),  Empty, Empty,
                    Value(8),  Value(8),  Empty, Empty,
                    Value(16), Value(16), Empty, Empty,
                ],
                direction: Direction::Left,
                expected_tiles: [
                    Value(4),  Empty, Empty, Empty,
                    Value(8),  Empty, Empty, Empty,
                    Value(16), Empty, Empty, Empty,
                    Value(32), Empty, Empty, Value(2),
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 15,
            },
            // Merge right
            MergeConfiguration {
                initial_tiles: [
                    Empty, Empty, Value(2),  Value(2),  
                    Empty, Empty, Value(4),  Value(4),  
                    Empty, Empty, Value(8),  Value(8),  
                    Empty, Empty, Value(16), Value(16), 
                ],
                direction: Direction::Right,
                expected_tiles: [
                    Empty,    Empty, Empty, Value(4),  
                    Empty,    Empty, Empty, Value(8),  
                    Empty,    Empty, Empty, Value(16), 
                    Value(2), Empty, Empty, Value(32), 
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 12,
            },
        ];

        for config in configurations {
            let mut board = Board::new_with_tiles(
                config.initial_tiles,
                FakeGameRng::new(2, config.new_tile_index),
            );
            match config.direction {
                Direction::Up => board.move_up(),
                Direction::Down => board.move_down(),
                Direction::Left => board.move_left(),
                Direction::Right => board.move_right(),
            }
            assert_eq!(
                board.tiles, config.expected_tiles,
                "Merge failed in direction {:?}",
                config.direction
            );
            assert_eq!(
                board.score, config.expected_score,
                "Incorrect score after the merge in direction {:?}",
                config.direction
            );
        }
    }

    #[test]
    fn merging_3_tiles() {
        #[rustfmt::skip]
        let configurations: [MergeConfiguration; _] = [
            // Merge up
            MergeConfiguration {
                initial_tiles: [
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Empty,    Empty,    Empty,    Empty,
                ],
                direction: Direction::Up,
                expected_tiles: [
                    Value(4), Value(8), Value(16), Value(32),
                    Value(2), Value(4), Value(8),  Value(16),
                    Empty,    Empty,    Empty,     Empty,
                    Empty,    Empty,    Empty,     Value(2),
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 15,
            },
            // Merge down
            MergeConfiguration {
                initial_tiles: [
                    Empty,    Empty,    Empty,    Empty,
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                ],
                direction: Direction::Down,
                expected_tiles: [
                    Empty,    Empty,    Empty,     Value(2),
                    Empty,    Empty,    Empty,     Empty,
                    Value(2), Value(4), Value(8),  Value(16),
                    Value(4), Value(8), Value(16), Value(32),
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 3,
            },
            // Merge left
            MergeConfiguration {
                initial_tiles: [
                    Value(2),  Value(2),  Value(2),  Empty,
                    Value(4),  Value(4),  Value(4),  Empty,
                    Value(8),  Value(8),  Value(8),  Empty,
                    Value(16), Value(16), Value(16), Empty,
                ],
                direction: Direction::Left,
                expected_tiles: [
                    Value(4),  Value(2),  Empty, Empty,
                    Value(8),  Value(4),  Empty, Empty,
                    Value(16), Value(8),  Empty, Empty,
                    Value(32), Value(16), Empty, Value(2),
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 15,
            },
            // Merge right
            MergeConfiguration {
                initial_tiles: [
                    Empty, Value(2),  Value(2),  Value(2),  
                    Empty, Value(4),  Value(4),  Value(4),  
                    Empty, Value(8),  Value(8),  Value(8),  
                    Empty, Value(16), Value(16), Value(16), 
                ],
                direction: Direction::Right,
                expected_tiles: [
                    Empty,    Empty, Value(2),  Value(4),  
                    Empty,    Empty, Value(4),  Value(8),  
                    Empty,    Empty, Value(8),  Value(16), 
                    Value(2), Empty, Value(16), Value(32), 
                ],
                expected_score: 4 + 8 + 16 + 32,
                new_tile_index: 12,
            },
        ];

        for config in configurations {
            let mut board = Board::new_with_tiles(
                config.initial_tiles,
                FakeGameRng::new(2, config.new_tile_index),
            );
            match config.direction {
                Direction::Up => board.move_up(),
                Direction::Down => board.move_down(),
                Direction::Left => board.move_left(),
                Direction::Right => board.move_right(),
            }
            assert_eq!(
                board.tiles, config.expected_tiles,
                "Merge failed in direction {:?}",
                config.direction
            );
            assert_eq!(
                board.score, config.expected_score,
                "Incorrect score after the merge in direction {:?}",
                config.direction
            );
        }
    }

    #[test]
    fn merging_4_tiles() {
        #[rustfmt::skip]
        let configurations: [MergeConfiguration; _] = [
            // Merge up
            MergeConfiguration {
                initial_tiles: [
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                ],
                direction: Direction::Up,
                expected_tiles: [
                    Value(4), Value(8), Value(16), Value(32),
                    Value(4), Value(8), Value(16), Value(32),
                    Empty,    Empty,    Empty,     Empty,
                    Empty,    Empty,    Empty,     Value(2),
                ],
                expected_score: 2 * (4 + 8 + 16 + 32),
                new_tile_index: 15,
            },
            // Merge down
            MergeConfiguration {
                initial_tiles: [
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                    Value(2), Value(4), Value(8), Value(16),
                ],
                direction: Direction::Down,
                expected_tiles: [
                    Empty,    Empty,    Empty,     Value(2),
                    Empty,    Empty,    Empty,     Empty,
                    Value(4), Value(8), Value(16), Value(32),
                    Value(4), Value(8), Value(16), Value(32),
                ],
                expected_score: 2 * (4 + 8 + 16 + 32),
                new_tile_index: 3,
            },
            // Merge left
            MergeConfiguration {
                initial_tiles: [
                    Value(2),  Value(2),  Value(2),  Value(2),  
                    Value(4),  Value(4),  Value(4),  Value(4),  
                    Value(8),  Value(8),  Value(8),  Value(8),  
                    Value(16), Value(16), Value(16), Value(16), 
                ],
                direction: Direction::Left,
                expected_tiles: [
                    Value(4),  Value(4),  Empty, Empty,
                    Value(8),  Value(8),  Empty, Empty,
                    Value(16), Value(16), Empty, Empty,
                    Value(32), Value(32), Empty, Value(2),
                ],
                expected_score: 2 * (4 + 8 + 16 + 32),
                new_tile_index: 15,
            },
            // Merge right
            MergeConfiguration {
                initial_tiles: [
                    Value(2),  Value(2),  Value(2),  Value(2),  
                    Value(4),  Value(4),  Value(4),  Value(4),  
                    Value(8),  Value(8),  Value(8),  Value(8),  
                    Value(16), Value(16), Value(16), Value(16), 
                ],
                direction: Direction::Right,
                expected_tiles: [
                    Empty,    Empty, Value(4),  Value(4),  
                    Empty,    Empty, Value(8),  Value(8),  
                    Empty,    Empty, Value(16), Value(16), 
                    Value(2), Empty, Value(32), Value(32), 
                ],
                expected_score: 2 * (4 + 8 + 16 + 32),
                new_tile_index: 12,
            },
        ];

        for config in configurations {
            let mut board = Board::new_with_tiles(
                config.initial_tiles,
                FakeGameRng::new(2, config.new_tile_index),
            );
            match config.direction {
                Direction::Up => board.move_up(),
                Direction::Down => board.move_down(),
                Direction::Left => board.move_left(),
                Direction::Right => board.move_right(),
            }
            assert_eq!(
                board.tiles, config.expected_tiles,
                "Merge failed in direction {:?}",
                config.direction
            );
            assert_eq!(
                board.score, config.expected_score,
                "Incorrect score after the merge in direction {:?}",
                config.direction
            );
        }
    }

    #[test]
    fn new_board_is_not_over() {
        let board = Board::new();
        assert_eq!(board.is_over(), false);
    }

    #[test]
    fn full_board_with_no_possible_move_is_over() {
        #[rustfmt::skip]
        let board = Board::new_with_tiles([
            Value(2), Value(4), Value(2), Value(4),
            Value(4), Value(2), Value(4), Value(2),
            Value(2), Value(4), Value(2), Value(4),
            Value(4), Value(2), Value(4), Value(2),
        ], FakeGameRng::new(2, 2));
        assert_eq!(board.is_over(), true);
    }

    #[test]
    fn full_board_with_possible_moves_is_not_over() {
        #[rustfmt::skip]
        let board = Board::new_with_tiles([
            Value(2), Value(4), Value(2), Value(4),
            Value(2), Value(2), Value(4), Value(2),
            Value(8), Value(4), Value(2), Value(8),
            Value(4), Value(2), Value(4), Value(4),
        ], FakeGameRng::new(2, 2));
        assert_eq!(board.is_over(), false);
    }

    #[test]
    fn can_move_up() {
        #[rustfmt::skip]
        let configurations: Vec<[Tile; NUM_TILES]> = vec![
            [
                Value(2), Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
            ], [
                Empty,    Empty, Empty, Empty,
                Value(2), Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
            ], [
                Value(2), Empty, Empty, Empty,
                Value(2), Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
            ],
        ];

        let expected_results = vec![false, true, true];

        for (index, example) in configurations.into_iter().zip(expected_results).enumerate() {
            let board = Board::new_with_tiles(example.0, FakeGameRng::new(2, 2));
            assert_eq!(
                board.can_move_up(),
                example.1,
                "Test failed for configuration {}",
                index + 1
            );
        }
    }

    #[test]
    fn can_move_down() {
        #[rustfmt::skip]
        let configurations: Vec<[Tile; NUM_TILES]> = vec![
            [
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Value(2), Empty, Empty, Empty,
            ], [
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Value(2), Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
            ], [
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Value(2), Empty, Empty, Empty,
                Value(2), Empty, Empty, Empty,
            ],
        ];

        let expected_results = vec![false, true, true];

        for (index, example) in configurations.into_iter().zip(expected_results).enumerate() {
            let board = Board::new_with_tiles(example.0, FakeGameRng::new(2, 2));
            assert_eq!(
                board.can_move_down(),
                example.1,
                "Test failed for configuration {}",
                index + 1
            );
        }
    }

    #[test]
    fn can_move_left() {
        #[rustfmt::skip]
        let configurations: Vec<[Tile; NUM_TILES]> = vec![
            [
                Value(2), Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
                Empty,    Empty, Empty, Empty,
            ], [
                Empty, Value(2), Empty, Empty,
                Empty, Empty,    Empty, Empty,
                Empty, Empty,    Empty, Empty,
                Empty, Empty,    Empty, Empty,
            ], [
                Value(2), Value(2), Empty, Empty,
                Empty,    Empty,    Empty, Empty,
                Empty,    Empty,    Empty, Empty,
                Empty,    Empty,    Empty, Empty,
            ],
        ];

        let expected_results = vec![false, true, true];

        for (index, example) in configurations.into_iter().zip(expected_results).enumerate() {
            let board = Board::new_with_tiles(example.0, FakeGameRng::new(2, 2));
            assert_eq!(
                board.can_move_left(),
                example.1,
                "Test failed for configuration {}",
                index + 1
            );
        }
    }

    #[test]
    fn can_move_right() {
        #[rustfmt::skip]
        let configurations: Vec<[Tile; NUM_TILES]> = vec![
            [
                Empty, Empty, Empty, Value(2),
                Empty, Empty, Empty, Empty,
                Empty, Empty, Empty, Empty,
                Empty, Empty, Empty, Empty,
            ], [
                Empty, Empty, Value(2), Empty,
                Empty, Empty, Empty,    Empty,
                Empty, Empty, Empty,    Empty,
                Empty, Empty, Empty,    Empty,
            ], [
                Empty, Empty, Value(2), Value(2),
                Empty, Empty, Empty,    Empty,
                Empty, Empty, Empty,    Empty,
                Empty, Empty, Empty,    Empty,
            ],
        ];

        let expected_results = vec![false, true, true];

        for (index, example) in configurations.into_iter().zip(expected_results).enumerate() {
            let board = Board::new_with_tiles(example.0, FakeGameRng::new(2, 2));
            assert_eq!(
                board.can_move_right(),
                example.1,
                "Test failed for configuration {}",
                index + 1
            );
        }
    }
}
