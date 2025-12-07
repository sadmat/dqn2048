use crate::dqn::stats::StatsRecorderType;
use crate::game::board::Board;
use crate::game::game_rng::RealGameRng;

#[derive(Debug)]
pub(crate) struct TrainingStats {
    pub epochs: usize,
    pub cumulated_epoch_rewards: f32,
    pub last_epoch_score: u32,
    pub best_tile: u32,
    pub replay_buffer_size: usize,
    pub epsilon: f64,
}

#[derive(Default)]
pub(crate) struct TrainingStatsRecorder {
    epoch_number: usize,
    reward_accumulator: f32,
    last_epoch_score: u32,
    best_tile: u32,
    replay_buffer_size: usize,
    epsilon: f64,
}

impl StatsRecorderType for TrainingStatsRecorder {
    type Stats = TrainingStats;
    type State = Board<RealGameRng>;

    fn record_new_epoch(&mut self) {
        self.epoch_number += 1;
        self.reward_accumulator = 0.0;
        self.last_epoch_score = 0;
        self.best_tile = 0;
        self.epsilon = 0.0;
    }

    fn record_reward(&mut self, reward: f32) {
        self.reward_accumulator += reward;
    }

    fn record_final_state(&mut self, state: &Self::State) {
        self.last_epoch_score = state.score;
        self.best_tile = state.max_tile_value();
    }

    fn record_replay_buffer_size(&mut self, size: usize) {
        self.replay_buffer_size = size;
    }
    fn record_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    fn stats(&self) -> Self::Stats {
        TrainingStats {
            epochs: self.epoch_number,
            cumulated_epoch_rewards: self.reward_accumulator,
            last_epoch_score: self.last_epoch_score,
            best_tile: self.best_tile,
            replay_buffer_size: self.replay_buffer_size,
            epsilon: self.epsilon,
        }
    }
}
