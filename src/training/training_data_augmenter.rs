use crate::dqn::data_augmenter::DataAugmenterType;
use crate::dqn::replay_buffer::StateTransition;
use crate::dqn::state::StateType;
use crate::game::board::Tile::{Empty, Value};
use crate::game::board::{Board, Direction, Tile, NUM_COLUMNS, NUM_ROWS, NUM_TILES};
use crate::game::game_rng::{GameRng, RealGameRng};

#[derive(Default)]
pub(crate) struct TrainingDataAugmenter {}

impl DataAugmenterType for TrainingDataAugmenter {
    type State = Board<RealGameRng>;

    fn augment(
        &self,
        state: Self::State,
        action: <Self::State as StateType>::Action,
        reward: f32,
        next_state: Self::State,
    ) -> Vec<StateTransition> {
        let mut state = state;
        let mut action = action;
        let mut next_state = next_state;
        let mut transitions = Vec::with_capacity(8);

        transitions.push(StateTransition::new(
            state.clone(),
            action,
            reward,
            next_state.clone(),
        ));
        transitions.push(StateTransition::new(
            state.mirrored_horizontally(),
            action.mirrored_horizontally(),
            reward,
            next_state.mirrored_horizontally(),
        ));

        for _ in 0..3 {
            state = state.rotated_cw();
            action = action.rotated_cw();
            next_state = next_state.rotated_cw();

            transitions.push(StateTransition::new(
                state.clone(),
                action,
                reward,
                next_state.clone(),
            ));
            transitions.push(StateTransition::new(
                state.mirrored_horizontally(),
                action.mirrored_horizontally(),
                reward,
                next_state.mirrored_horizontally(),
            ));
        }

        transitions
    }
}

impl<R: GameRng> Board<R> {
    fn mirrored_horizontally(&self) -> Self {
        let mut tiles = [Tile::Empty; NUM_TILES];
        let new_index = |row, column| row * NUM_ROWS + NUM_COLUMNS - 1 - column;

        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                tiles[new_index(row, column)] = self
                    .value_at(row, column)
                    .map_or_else(|| Empty, |value| Value(value));
            }
        }
        Board::new_with_tiles_and_score(tiles, self.score, R::default())
    }

    fn rotated_cw(&self) -> Self {
        let mut tiles = [Tile::Empty; NUM_TILES];
        let new_index = |row, column| NUM_COLUMNS * column + NUM_ROWS - row - 1;

        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                tiles[new_index(row, column)] = self
                    .value_at(row, column)
                    .map_or_else(|| Empty, |value| Value(value));
            }
        }
        Board::new_with_tiles_and_score(tiles, self.score, R::default())
    }
}

impl Direction {
    fn mirrored_horizontally(&self) -> Self {
        match self {
            Direction::Up => Direction::Up,
            Direction::Down => Direction::Down,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    fn rotated_cw(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
            Direction::Right => Direction::Down,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Tile::{Empty, Value};
    use crate::game::board::{Direction, Tile, NUM_TILES};

    #[test]
    fn board_mirroring() {
        #[rustfmt::skip]
        let original_states: Vec<[Tile; NUM_TILES]> = vec![
            [
                Value(2), Value(4),  Empty, Empty,
                Value(8), Value(16), Empty, Empty,
                Empty,    Empty,     Empty, Empty,
                Empty,    Empty,     Empty, Empty,
            ], [
                Empty, Empty, Value(4),  Value(2),
                Empty, Empty, Value(16), Value(8),
                Empty, Empty, Empty, Empty,
                Empty, Empty, Empty, Empty,
            ], [
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Value(16), Value(8),
                Empty, Empty, Value(4),  Value(2),
            ], [
                Empty, Empty,        Empty, Empty,
                Empty, Empty,        Empty, Empty,
                Value(8), Value(16), Empty, Empty,
                Value(2), Value(4),  Empty, Empty,
            ],
        ];

        #[rustfmt::skip]
        let mirrored_states: Vec<[Tile; NUM_TILES]> = vec![
            [
                Empty, Empty, Value(4),  Value(2),
                Empty, Empty, Value(16), Value(8),
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Empty,     Empty,
            ], [
                Value(2), Value(4),  Empty, Empty,
                Value(8), Value(16), Empty, Empty,
                Empty,    Empty,     Empty, Empty,
                Empty,    Empty,     Empty, Empty,
            ], [
                Empty,    Empty,     Empty, Empty,
                Empty,    Empty,     Empty, Empty,
                Value(8), Value(16), Empty, Empty,
                Value(2), Value(4),  Empty, Empty,
            ], [
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Value(16), Value(8),
                Empty, Empty, Value(4),  Value(2),
            ],
        ];

        for (index, (original, mirrored)) in
            original_states.into_iter().zip(mirrored_states).enumerate()
        {
            let original_board = Board::new_with_tiles(original, RealGameRng::new());
            let mirrored_board = Board::new_with_tiles(mirrored, RealGameRng::new());
            assert_eq!(
                original_board.mirrored_horizontally(),
                mirrored_board,
                "Mismatch at index {}",
                index
            );
        }
    }

    #[test]
    fn board_rotating() {
        #[rustfmt::skip]
        let original_states: Vec<[Tile; NUM_TILES]> = vec![
            [
                Value(2), Value(4),  Empty, Empty,
                Value(8), Value(16), Empty, Empty,
                Empty,    Empty,     Empty, Empty,
                Empty,    Empty,     Empty, Empty,
            ], [
                Empty, Empty, Value(4),  Value(2),
                Empty, Empty, Value(16), Value(8),
                Empty, Empty, Empty, Empty,
                Empty, Empty, Empty, Empty,
            ], [
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Value(16), Value(8),
                Empty, Empty, Value(4),  Value(2),
            ], [
                Empty, Empty,        Empty, Empty,
                Empty, Empty,        Empty, Empty,
                Value(8), Value(16), Empty, Empty,
                Value(2), Value(4),  Empty, Empty,
            ],
        ];

        #[rustfmt::skip]
        let rotated_states: Vec<[Tile; NUM_TILES]> = vec![
            [
                Empty, Empty, Value(8),  Value(2),
                Empty, Empty, Value(16), Value(4),
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Empty,     Empty,
            ], [
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Empty,     Empty,
                Empty, Empty, Value(16), Value(4),
                Empty, Empty, Value(8),  Value(2),
            ], [
                Empty,    Empty,     Empty, Empty,
                Empty,    Empty,     Empty, Empty,
                Value(4), Value(16), Empty, Empty,
                Value(2), Value(8),  Empty, Empty,
            ], [
                Value(2), Value(8),  Empty, Empty,
                Value(4), Value(16), Empty, Empty,
                Empty,    Empty,     Empty, Empty,
                Empty,    Empty,     Empty, Empty,
            ]
        ];

        for (index, (original, rotated)) in
            original_states.into_iter().zip(rotated_states).enumerate()
        {
            let original_board = Board::new_with_tiles(original, RealGameRng::new());
            let rotated_board = Board::new_with_tiles(rotated, RealGameRng::new());
            assert_eq!(
                original_board.rotated_cw(),
                rotated_board,
                "Mismatch at index {}",
                index
            );
        }
    }

    #[test]
    fn direction_mirroring() {
        let original_directions = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];
        let mirrored_directions = [
            Direction::Up,
            Direction::Down,
            Direction::Right,
            Direction::Left,
        ];
        for (original, mirrored) in original_directions.into_iter().zip(mirrored_directions) {
            assert_eq!(original.mirrored_horizontally(), mirrored);
        }
    }

    #[test]
    fn direction_rotating() {
        let original_directions = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];
        let mirrored_directions = [
            Direction::Right,
            Direction::Left,
            Direction::Up,
            Direction::Down,
        ];
        for (original, mirrored) in original_directions.into_iter().zip(mirrored_directions) {
            assert_eq!(original.rotated_cw(), mirrored);
        }
    }
}
