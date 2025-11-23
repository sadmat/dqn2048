use slint::JoinHandle;

use crate::{PlotSize, training::training_stats_recorder::TrainingStats};

use std::sync::mpsc::*;

pub(crate) struct PlotsSizes {
    pub score_plot_size: PlotSize,
    pub reward_plot_size: PlotSize,
    pub best_tile_plot_size: PlotSize,
}

pub(crate) enum TrainingOverviewUpdate {
    EpochFinished(TrainingStats),
    PlotsSizesChanged(PlotsSizes),
}

pub(crate) struct TrainingOverviewThread {
    scores: Vec<u32>,
    rewards: Vec<f32>,
    best_tiles: Vec<u32>,
    best_score: u32,
    best_tile: u32,
}

impl TrainingOverviewThread {
    pub(crate) fn spawn_thread(update_channel: Receiver<TrainingOverviewUpdate>) -> JoinHandle<()> {
        todo!()
    }
}
