use crate::dqn::stats::StatsRecorderType;
use crate::game::board::Board;
use crate::game::game_rng::RealGameRng;

pub(crate) struct TrainingStats {
    pub epochs: usize,
    pub cumulated_epoch_rewards: f32,
    pub last_epoch_score: u32,
}

#[derive(Default)]
pub(crate) struct TrainingStatsRecorder {
    epoch_number: usize,
    reward_accumulator: f32,
    last_epoch_score: u32,
}

impl StatsRecorderType for TrainingStatsRecorder {
    type Stats = TrainingStats;
    type State = Board<RealGameRng>;

    fn record_new_epoch(&mut self) {
        self.epoch_number += 1;
        self.reward_accumulator = 0.0;
        self.last_epoch_score = 0;
    }

    fn record_reward(&mut self, reward: f32) {
        self.reward_accumulator += reward;
    }

    fn record_final_state(&mut self, state: &Self::State) {
        self.last_epoch_score = state.score;
    }

    fn stats(&self) -> Self::Stats {
        TrainingStats {
            epochs: self.epoch_number,
            cumulated_epoch_rewards: self.reward_accumulator,
            last_epoch_score: self.last_epoch_score,
        }
    }
}