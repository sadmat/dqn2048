use crate::training::training_stats_recorder::TrainingStats;

#[derive(PartialEq)]
pub(crate) enum TrainingState {
    Idle,
    Training,
}

pub(crate) enum TrainingAction {
    Start,
    Pause,
}

pub(crate) enum TrainingMessage {
    StateChanged(TrainingState),
    EpochFinished(TrainingStats),
}

