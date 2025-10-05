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

#[derive(PartialEq, Copy, Clone, Debug)]
enum VerticalDirection {
    Up,
    Down,
}

#[derive(PartialEq, Copy, Clone, Debug)]
enum HorizontalDirection {
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
        self.move_tiles(Direction::Right);
    }

    pub fn move_left(&mut self) {
        self.move_tiles(Direction::Left);
    }

    pub fn move_down(&mut self) {
        self.move_tiles(Direction::Down);
    }

    pub fn move_up(&mut self) {
        self.move_tiles(Direction::Up);
    }

    fn move_tiles(&mut self, direction: Direction) {
        println!("entering move_tiles({:?})", direction);
        io::stdout().flush().unwrap();
        let mut merged = [false; NUM_TILES];
        if let Some(direction) = direction.horizontal_direction() {
            let mut column = direction.start_from();
            while (0..NUM_COLUMNS).contains(&column) {
                self.move_column(column, direction, &mut merged);
                column = direction.next_column(column);
            }
        } else if let Some(direction) = direction.vertical_direction() {
            let mut row = direction.start_from();
            while (0..NUM_ROWS).contains(&row) {
                self.move_row(row, direction, &mut merged);
                row = direction.next_row(row);
            }
        }
        println!("move_tiles({:?}): before place_random_tile", direction);
        io::stdout().flush().unwrap();
        self.place_random_tile();
        println!("exiting move_tiles({:?})", direction);
        io::stdout().flush().unwrap();
    }

    fn move_column(
        &mut self,
        column: usize,
        direction: HorizontalDirection,
        merged_tiles: &mut [bool; NUM_TILES],
    ) {
        if direction == HorizontalDirection::Left && column == 0 { return; }
        if direction == HorizontalDirection::Right && column == NUM_COLUMNS - 1 { return; }

        'rows: for row in 0..NUM_ROWS {
            let source_index = index(row, column);
            let Tile::Value(source_value) = self.tiles[source_index] else {
                continue;
            };

            let mut last_empty_column = column;
            let mut target_column = direction.next_column(column);
            while (0..NUM_COLUMNS).contains(&target_column) {
                let target_index = index(row, target_column);

                if self.tiles[target_index] == Tile::Empty {
                    last_empty_column = target_column;
                } else if let Tile::Value(target_value) = self.tiles[target_index] {
                    // Can we merge the tiles?
                    if source_value == target_value && !merged_tiles[target_index] {
                        self.tiles[source_index] = Tile::Empty;
                        self.tiles[target_index] = Tile::Value(source_value + target_value);
                        self.score += source_value + target_value;
                        merged_tiles[target_index] = true;
                        continue 'rows;
                    }
                    // No more empty fields
                    break;
                }
                target_column = direction.next_column(target_column);
            }

            let target_index = index(row, last_empty_column);
            if target_index != source_index {
                self.tiles[source_index] = Tile::Empty;
                self.tiles[target_index] = Tile::Value(source_value);
            }
        }
    }

    fn move_row(
        &mut self,
        row: usize,
        direction: VerticalDirection,
        merged_tiles: &mut [bool; NUM_TILES],
    ) {
        if direction == VerticalDirection::Up && row == 0 { return; }
        if direction == VerticalDirection::Down && row == NUM_ROWS - 1 { return; }

        'columns: for column in 0..NUM_COLUMNS {
            let source_index = index(row, column);
            let Tile::Value(source_value) = self.tiles[source_index] else {
                continue;
            };

            let mut last_empty_row = row;
            let mut target_row = direction.next_row(row);
            while (0..NUM_ROWS).contains(&target_row) {
                let target_index = index(target_row, column);

                if self.tiles[target_index] == Tile::Empty {
                    last_empty_row = target_row;
                } else if let Tile::Value(target_value) = self.tiles[target_index] {
                    // Can we merge the tiles?
                    if source_value == target_value && !merged_tiles[target_index] {
                        self.tiles[source_index] = Tile::Empty;
                        self.tiles[target_index] = Tile::Value(source_value + target_value);
                        self.score += source_value + target_value;
                        merged_tiles[target_index] = true;
                        continue 'columns;
                    }
                    // No more empty fields
                    break;
                }
                target_row = direction.next_row(target_row);
            }

            let target_index = index(last_empty_row, column);
            if target_index != source_index {
                self.tiles[source_index] = Tile::Empty;
                self.tiles[target_index] = Tile::Value(source_value);
            }
        }
    }
}

fn index(row: usize, column: usize) -> usize {
    row * NUM_ROWS + column
}

impl Direction {
    fn vertical_direction(&self) -> Option<VerticalDirection> {
        match self {
            Direction::Up => Some(VerticalDirection::Up),
            Direction::Down => Some(VerticalDirection::Down),
            _ => None,
        }
    }

    fn horizontal_direction(&self) -> Option<HorizontalDirection> {
        match self {
            Direction::Left => Some(HorizontalDirection::Left),
            Direction::Right => Some(HorizontalDirection::Right),
            _ => None,
        }
    }
}

impl VerticalDirection {
    fn start_from(&self) -> usize {
        match self {
            VerticalDirection::Up => 0,
            VerticalDirection::Down => (NUM_ROWS - 1).try_into().unwrap(),
        }
    }

    fn delta(&self) -> isize {
        match self {
            VerticalDirection::Up => 1,
            VerticalDirection::Down => -1,
        }
    }

    fn next_row(&self, row: usize) -> usize {
        (row as i32 + self.delta() as i32) as usize
    }
}

impl HorizontalDirection {
    fn start_from(&self) -> usize {
        match self {
            HorizontalDirection::Left => 0,
            HorizontalDirection::Right => (NUM_COLUMNS - 1).try_into().unwrap(),
        }
    }

    fn delta(&self) -> isize {
        match self {
            HorizontalDirection::Left => 1,
            HorizontalDirection::Right => -1,
        }
    }

    fn next_column(&self, column: usize) -> usize {
        (column as i32 + self.delta() as i32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::Tile::*;
    use super::*;

    struct MergeConfiguration {
        initial_tiles: [Tile; NUM_TILES],
        direction: Direction,
        expected_tiles: [Tile; NUM_TILES],
        expected_score: u32,
        new_tile_index: usize,
    }

    // Tests:
    // [x] Board initialization
    // [x] Moves
    // [x] Random tile after the move
    // [ ] Merging two tiles
    // [ ] Merging two tiles when 3 tiles in row/column have the same value
    // [ ] Merging tiles when 4 tiles in row/column have the same value
    // [ ] Score when merging
    // [ ] Game over when board is filled and there is no move possible

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
}
