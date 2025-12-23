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

impl StateType for Board<RealGameRng> {
    type Action = Direction;

    const NUM_ACTIONS: usize = 4;
    const NUM_FEATURES: usize = NUM_TILES * 12;

    fn initial_state() -> Board<RealGameRng> {
        Board::new()
    }

    fn possible_actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::with_capacity(Self::NUM_ACTIONS);

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
            Direction::Up => board.move_up(),
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
        // TODO: Fill existing buffer instead of creating new one
        let mut features = Vec::with_capacity(Self::NUM_FEATURES);
        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                let tile_value = self.value_at(row, column).unwrap_or_default();
                let mut inputs = vec![0.0; 12];
                if tile_value > 0 {
                    let index = tile_value.ilog2().min((inputs.len() - 1) as u32) as usize;
                    inputs[index] = 1.0;
                }
                features.extend(inputs);
            }
        }
        features
    }
}
