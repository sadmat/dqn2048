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
    EpochFinished(EpochStats),
}

pub(crate) struct EpochStats {
    pub epoch_score: u32,
    pub highest_tile_value: u32,
    pub current_epoch: u32,
    pub epochs_per_second: f32,
}
