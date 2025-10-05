use rand::prelude::*;

#[derive(Copy, Clone, PartialEq, Debug)]
pub(crate) enum Tile {
    Empty,
    Value(u32),
}

const NUM_ROWS: usize = 4;
const NUM_COLUMNS: usize = 4;
const NUM_TILES: usize = NUM_ROWS * NUM_COLUMNS;

pub(crate) struct Board {
    score: u32,
    tiles: [Tile; NUM_TILES],
}

impl Board {
    pub fn new() -> Board {
        let mut board = Board {
            score: 0,
            tiles: [Tile::Empty; NUM_TILES],
        };
        board.place_random_tile();
        board.place_random_tile();
        board
    }

    fn new_with_tiles(tiles: [Tile; NUM_TILES]) -> Board {
        Board {
            score: 0,
            tiles: tiles,
        }
    }

    fn place_random_tile(&mut self) {
        // TODO: check for full board
        let mut rng = rand::rng();
        let value = if rng.random::<f64>() <= 0.1 { 4 } else { 2 };
        loop {
            let index = rng.random_range(0..NUM_TILES);
            if self.tiles[index] == Tile::Empty {
                self.tiles[index] = Tile::Value(value);
                break;
            }
        }
    }

    pub fn move_right(&mut self) {
        for row in 0..NUM_ROWS {
            for source_column in (0..NUM_COLUMNS - 1).rev() {
                let Tile::Value(value) = self.tiles[row * NUM_ROWS + source_column] else {
                    continue;
                };

                for destination_column in (source_column + 1..NUM_COLUMNS).rev() {
                    if self.tiles[row * NUM_ROWS + destination_column] != Tile::Empty {
                        continue;
                    }
                    self.tiles[row * NUM_ROWS + destination_column] = Tile::Value(value);
                    self.tiles[row * NUM_ROWS + source_column] = Tile::Empty;
                    break;
                }
            }
        }
    }

    pub fn move_left(&mut self) {
        for row in 0..NUM_ROWS {
            for source_column in (1..NUM_COLUMNS) {
                let Tile::Value(value) = self.tiles[row * NUM_ROWS + source_column] else {
                    continue;
                };

                for destination_column in (0..source_column) {
                    if self.tiles[row * NUM_ROWS + destination_column] != Tile::Empty {
                        continue;
                    }
                    self.tiles[row * NUM_ROWS + destination_column] = Tile::Value(value);
                    self.tiles[row * NUM_ROWS + source_column] = Tile::Empty;
                    break;
                }
            }
        }
    }

    pub fn move_down(&mut self) {
        for column in 0..NUM_COLUMNS {
            for source_row in (0..NUM_ROWS - 1).rev() {
                let Tile::Value(value) = self.tiles[source_row * NUM_ROWS + column] else {
                    continue;
                };

                for destination_row in (source_row + 1..NUM_ROWS).rev() {
                    if self.tiles[destination_row * NUM_ROWS + column] != Tile::Empty {
                        continue;
                    }
                    self.tiles[destination_row * NUM_ROWS + column] = Tile::Value(value);
                    self.tiles[source_row * NUM_ROWS + column] = Tile::Empty;
                    break;
                }
            }
        }
    }

    pub fn move_up(&mut self) {
        for column in 0..NUM_COLUMNS {
            for source_row in (1..NUM_ROWS) {
                let Tile::Value(value) = self.tiles[source_row * NUM_ROWS + column] else {
                    continue;
                };

                for destination_row in (0..NUM_ROWS) {
                    if self.tiles[destination_row * NUM_ROWS + column] != Tile::Empty {
                        continue;
                    }
                    self.tiles[destination_row * NUM_ROWS + column] = Tile::Value(value);
                    self.tiles[source_row * NUM_ROWS + column] = Tile::Empty;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests:
    // [x] Board initialization
    // [ ] Moves
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
        use super::Tile::*;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Value(2), Empty,    Empty,    Empty,
            Value(2), Value(4), Empty,    Empty,
            Value(2), Value(4), Value(8), Empty,
            Value(2), Value(4), Value(8), Value(16),
        ]);

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Empty,    Empty,    Empty,    Value(2), 
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Value(2), Value(4), Value(8),
            Value(2), Value(4), Value(8), Value(16),
        ];

        board.move_right();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn move_left() {
        use super::Tile::*;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Empty,    Empty,    Empty,    Value(2), 
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Value(2), Value(4), Value(8),
            Value(2), Value(4), Value(8), Value(16),
        ]);

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Value(2), Empty,    Empty,    Empty,
            Value(2), Value(4), Empty,    Empty,
            Value(2), Value(4), Value(8), Empty,
            Value(2), Value(4), Value(8), Value(16),
        ];

        board.move_left();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn move_down() {
        use super::Tile::*;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Value(2), Value(4), Value(8), Value(16),
            Empty,    Value(2), Value(4), Value(8),
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Empty,    Empty,    Value(2), 
        ]);

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Empty,    Empty,    Empty,     Value(16),
            Empty,    Empty,    Value(8),  Value(8),
            Empty,    Value(4), Value(4),  Value(4), 
            Value(2), Value(2), Value(2),  Value(2), 
        ];

        board.move_down();

        assert_eq!(board.tiles, expected_result);
    }

    #[test]
    fn move_up() {
        use super::Tile::*;

        #[rustfmt::skip]
        let mut board = Board::new_with_tiles([
            Empty,    Empty,    Empty,     Value(16),
            Empty,    Empty,    Value(8),  Value(8),
            Empty,    Value(4), Value(4),  Value(4), 
            Value(2), Value(2), Value(2),  Value(2), 
        ]);

        #[rustfmt::skip]
        let expected_result: [Tile; _] = [
            Value(2), Value(4), Value(8), Value(16),
            Empty,    Value(2), Value(4), Value(8),
            Empty,    Empty,    Value(2), Value(4), 
            Empty,    Empty,    Empty,    Value(2), 
        ];

        board.move_up();

        assert_eq!(board.tiles, expected_result);
    }
}
