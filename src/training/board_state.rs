use crate::{
    dqn::state::{ActionType, StateType},
    game::{
        board::{Board, Direction, NUM_COLUMNS, NUM_ROWS, NUM_TILES},
        game_rng::RealGameRng,
    },
};

impl ActionType for Direction {
    fn index(&self) -> usize {
        match self {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        }
    }
}

const NUM_FEATURES: usize = NUM_TILES * 11;

impl StateType for Board<RealGameRng> {
    type Action = Direction;

    fn initial_state() -> Board<RealGameRng> {
        Board::new()
    }

    fn num_actions() -> usize {
        4
    }

    fn num_features() -> usize {
        NUM_FEATURES
    }

    fn possible_actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::with_capacity(Self::num_actions());

        if self.can_move_up() {
            actions.push(Direction::Up);
        }
        if self.can_move_down() {
            actions.push(Direction::Down);
        }
        if self.can_move_left() {
            actions.push(Direction::Left);
        }
        if self.can_move_right() {
            actions.push(Direction::Right);
        }

        actions
    }

    fn advance(&self, action: &Self::Action) -> Self {
        let mut board = self.clone();
        match action {
            Direction::Up => board.move_right(),
            Direction::Down => board.move_down(),
            Direction::Left => board.move_left(),
            Direction::Right => board.move_right(),
        }
        board
    }

    fn is_terminal(&self) -> bool {
        self.is_over()
    }

    fn as_features(&self) -> Vec<f32> {
        // TODO: one_hot tensor?
        let mut features = Vec::with_capacity(NUM_FEATURES);
        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                let tile_value = self.value_at(row, column).unwrap_or_default();
                #[rustfmt::skip]
                let inputs = match tile_value {
                    2    => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    4    => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    8    => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    16   => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    32   => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    64   => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    128  => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    256  => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    512  => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    1024 => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    2048 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    _    => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                };
                features.extend_from_slice(inputs.as_slice());
            }
        }
        features
    }
}
