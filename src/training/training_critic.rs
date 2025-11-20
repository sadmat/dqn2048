use crate::{
    dqn::{critic::CriticType, state::StateType},
    game::{board::Board, game_rng::RealGameRng},
};

pub(crate) struct TrainingCritic {}

impl TrainingCritic {
    pub(crate) fn new() -> Self {
        TrainingCritic {}
    }
}

impl CriticType for TrainingCritic {
    type State = Board<RealGameRng>;

    fn reward(
        &self,
        state: &Self::State,
        action: &<Self::State as StateType>::Action,
        next_state: &Self::State,
    ) -> f32 {
        if next_state.is_over() {
            -1.0
        } else {
            if next_state.score == 0 {
                0.0
            } else {
                // TODO: test score per move reward
                // (next_state.score - state.score) as f32 / next_state.score as f32
                (next_state.score - state.score) as f32 / next_state.max_tile_value() as f32
            }
        }
    }
}
