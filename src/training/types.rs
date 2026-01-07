use std::path::PathBuf;

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
    SaveModel(PathBuf),
    LoadModel(PathBuf),
    SaveSession(PathBuf),
    LoadSession(PathBuf),
}

#[derive(Debug)]
pub(crate) enum TrainingMessage {
    StateChanged(TrainingState),
    EpochFinished(TrainingStats),
}
