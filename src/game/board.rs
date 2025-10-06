use rand::distr::slice::Empty;
use rand::prelude::*;
use std::io;
use std::io::Write;

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

const NUM_ROWS: usize = 4;
const NUM_COLUMNS: usize = 4;
const NUM_TILES: usize = NUM_ROWS * NUM_COLUMNS;

pub(crate) struct Board<F1, F2>
where
    F1: FnMut() -> u32,
    F2: FnMut() -> usize,
{
    score: u32,
    tiles: [Tile; NUM_TILES],
    random_tile_value: F1,
    random_index: F2,
}

impl Board<fn() -> u32, fn() -> usize> {
    pub fn new() -> Board<impl FnMut() -> u32, impl FnMut() -> usize> {
        let mut tile_rng = rand::rng();
        let mut index_rng = rand::rng();

        let mut board = Board {
            score: 0,
            tiles: [Tile::Empty; NUM_TILES],
            random_tile_value: move || {
                if tile_rng.random::<f64>() <= 0.1 {
                    4
                } else {
                    2
                }
            },
            random_index: move || index_rng.random_range(0..NUM_TILES),
        };
        board.place_random_tile();
        board.place_random_tile();
        board
    }
}

impl<F1, F2> Board<F1, F2>
where
    F1: FnMut() -> u32,
    F2: FnMut() -> usize,
{
    fn new_with_tiles(
        tiles: [Tile; NUM_TILES],
        random_tile_value: F1,
        random_index: F2,
    ) -> Board<F1, F2> {
        Board {
            score: 0,
            tiles: tiles,
            random_tile_value: random_tile_value,
            random_index: random_index,
        }
    }

    fn place_random_tile(&mut self) {
        // TODO: check for full board
        loop {
            let index = (self.random_index)();
            println!("random tile index: {index}");
            if self.tiles[index] == Tile::Empty {
                self.tiles[index] = Tile::Value((self.random_tile_value)());
                break;
            }
        }
    }

    pub fn move_right(&mut self) {
        let mut merged_tiles = [false; NUM_TILES];
        for column in (0..NUM_COLUMNS - 1).rev() {
            for row in 0..NUM_ROWS {
                self.slide_tile(row, column, Direction::Right, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    pub fn move_left(&mut self) {
        let mut merged_tiles = [false; NUM_TILES];
        for column in 1..NUM_COLUMNS {
            for row in 0..NUM_ROWS {
                self.slide_tile(row, column, Direction::Left, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    pub fn move_down(&mut self) {
        let mut merged_tiles = [false; NUM_TILES];
        for row in (0..NUM_ROWS - 1).rev() {
            for column in 0..NUM_COLUMNS {
                self.slide_tile(row, column, Direction::Down, &mut merged_tiles);
            }
        }
        self.place_random_tile();
    }

    pub fn move_up(&mut self) {
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

    fn is_over(&self) -> bool {
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

    fn value_at(&self, row: usize, column: usize) -> Option<u32> {
        if !(0..NUM_ROWS).contains(&row) || !(0..NUM_COLUMNS).contains(&column) {
            return None;
        }

        match self.tiles[index(row, column)] {
            Tile::Empty => None,
            Tile::Value(value) => Some(value),
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
        let new_tile_value = || 2;
        let new_tile_index = || 0;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Value(2), Empty,    Empty,    Empty,
            Value(2), Value(4), Empty,    Empty,
            Value(2), Value(4), Value(8), Empty,
            Value(2), Value(4), Value(8), Value(16),
        ], new_tile_value, new_tile_index);

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
        let new_tile_value = || 2;
        let new_tile_index = || 3;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Empty,    Empty,    Empty,    Value(2), 
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Value(2), Value(4), Value(8),
            Value(2), Value(4), Value(8), Value(16),
        ], new_tile_value, new_tile_index);

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
        let new_tile_value = || 2;
        let new_tile_index = || 0;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Value(2), Value(4), Value(8), Value(16),
            Empty,    Value(2), Value(4), Value(8),
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Empty,    Empty,    Value(2), 
        ], new_tile_value, new_tile_index);

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
        let new_tile_value = || 2;
        let new_tile_index = || 12;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Empty,    Empty,    Empty,     Value(16),
            Empty,    Empty,    Value(8),  Value(8),
            Empty,    Value(4), Value(4),  Value(4), 
            Value(2), Value(2), Value(2),  Value(2), 
        ], new_tile_value, new_tile_index);

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
            let new_tile_index = || config.new_tile_index;
            let mut board = Board::new_with_tiles(config.initial_tiles, || 2, new_tile_index);
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
            let new_tile_index = || config.new_tile_index;
            let mut board = Board::new_with_tiles(config.initial_tiles, || 2, new_tile_index);
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
            let new_tile_index = || config.new_tile_index;
            let mut board = Board::new_with_tiles(config.initial_tiles, || 2, new_tile_index);
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
        ], || 2, || 2);
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
        ], || 2, || 2);
        assert_eq!(board.is_over(), false);
    }
}
