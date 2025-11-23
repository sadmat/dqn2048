use crate::{UiTrainingState, training::types::TrainingState};

impl TrainingState {
    pub(crate) fn as_ui_training_state(&self) -> UiTrainingState {
        match self {
            TrainingState::Idle => UiTrainingState::Idle,
            TrainingState::Training => UiTrainingState::Training,
        }
    }
}
