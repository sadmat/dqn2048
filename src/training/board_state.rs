use crate::{
    dqn::state::{ActionType, State},
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

impl State<NUM_TILES> for Board<RealGameRng> {
    type Action = Direction;

    fn initial_state() -> Board<RealGameRng> {
        Board::new()
    }

    fn num_actions() -> usize {
        4
    }

    fn possible_actions(&self) -> Vec<Self::Action> {
        todo!()
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

    fn as_features(&self) -> [f32; NUM_TILES] {
        todo!()
    }
}
