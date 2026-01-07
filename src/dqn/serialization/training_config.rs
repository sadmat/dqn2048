use serde::{Deserialize, Serialize};

use crate::dqn::trainer::Hyperparameters;

#[derive(Serialize, Deserialize)]
pub(crate) struct ReplayBufferConfig {
    pub capacity: usize,
    pub size: usize,
    pub write_position: usize,
}

impl ReplayBufferConfig {
    pub(crate) fn with(capacity: usize, size: usize, write_position: usize) -> Self {
        Self {
            capacity,
            size,
            write_position,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TrainingInfo {
    pub epoch_number: usize,
    pub frame_number: usize,
}

impl TrainingInfo {
    pub(crate) fn with(epoch_number: usize, frame_number: usize) -> Self {
        Self {
            epoch_number,
            frame_number,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TrainingConfig {
    pub hyperparameters: Hyperparameters,
    pub replay_buffer: ReplayBufferConfig,
    pub training_info: TrainingInfo,
}

impl TrainingConfig {
    pub(crate) fn with(
        hyperparameters: Hyperparameters,
        replay_buffer_config: ReplayBufferConfig,
        training_info: TrainingInfo,
    ) -> Self {
        Self {
            hyperparameters,
            replay_buffer: replay_buffer_config,
            training_info,
        }
    }
}
