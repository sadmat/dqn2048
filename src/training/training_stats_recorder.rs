use std::time::Instant;
use crate::dqn::stats::StatsRecorderType;
use crate::game::board::{Board, NUM_COLUMNS, NUM_ROWS};
use crate::game::game_rng::RealGameRng;

#[derive(Debug)]
pub(crate) struct TrainingStats {
    pub epochs: usize,
    pub epochs_per_second: Option<f32>,
    pub cumulated_epoch_rewards: f32,
    pub last_epoch_score: u32,
    pub best_tile: u32,
}

#[derive(Default)]
pub(crate) struct TrainingStatsRecorder {
    epoch_number: usize,
    epoch_timestamp: Option<(usize, Instant)>,
    reward_accumulator: f32,
    last_epoch_score: u32,
    best_tile: u32,
}

impl StatsRecorderType for TrainingStatsRecorder {
    type Stats = TrainingStats;
    type State = Board<RealGameRng>;

    fn record_new_epoch(&mut self) {
        self.epoch_number += 1;
        self.reward_accumulator = 0.0;
        self.last_epoch_score = 0;
        self.best_tile = 0;
        if self.epoch_timestamp.is_none() {
            self.epoch_timestamp = Some((self.epoch_number, Instant::now()));
        } else if let Some((_, timestamp)) = &self.epoch_timestamp && timestamp.elapsed().as_secs_f32() > 2.0 {
            self.epoch_timestamp = Some((self.epoch_number, Instant::now()));
        }
    }

    fn record_reward(&mut self, reward: f32) {
        self.reward_accumulator += reward;
    }

    fn record_final_state(&mut self, state: &Self::State) {
        self.last_epoch_score = state.score;
        for row in 0..NUM_ROWS {
            for column in 0..NUM_COLUMNS {
                if let Some(value) = state.value_at(row, column) {
                    self.best_tile = self.best_tile.max(value);
                }
            }
        }
    }

    fn stats(&self) -> Self::Stats {
        let epochs_per_second = self.epoch_timestamp
            .map(|(epochs_on_timestamp, timestamp)| {
                let epochs_since_timestamp = self.epoch_number - epochs_on_timestamp;
                let duration = timestamp.elapsed();
                epochs_since_timestamp as f32 / duration.as_secs_f32()
            });

        TrainingStats {
            epochs: self.epoch_number,
            epochs_per_second,
            cumulated_epoch_rewards: self.reward_accumulator,
            last_epoch_score: self.last_epoch_score,
            best_tile: self.best_tile,
        }
    }
}