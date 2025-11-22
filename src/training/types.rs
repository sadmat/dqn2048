use crate::training::training_stats_recorder::TrainingStats;

#[derive(Debug, PartialEq)]
pub(crate) enum TrainingState {
    Idle,
    Training,
}

#[derive(Debug)]
pub(crate) enum TrainingAction {
    Start,
    Pause,
}

#[derive(Debug)]
pub(crate) enum TrainingMessage {
    StateChanged(TrainingState),
    EpochFinished(TrainingStats),
}
