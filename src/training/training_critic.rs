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
            return -10.0;
        }

        let score_gain = (next_state.score.saturating_sub(state.score)) as f32;
        let score_reward = if score_gain > 0.0 {
            score_gain.log2() * 0.1
        } else { 0.0 };

        let empty_reward = next_state.num_empty_tiles() as f32 / 16.0;

        score_reward + empty_reward
    }
}

impl Board<RealGameRng> {
    fn num_empty_tiles(&self) -> usize {
        let mut empty = 0;
        for row in 0..4 {
            for column in 0..4 {
                if matches!(self.value_at(row, column), None) {
                    empty += 1;
                }
            }
        }
        empty
    }
}