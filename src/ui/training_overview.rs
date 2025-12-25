use plotters::prelude::*;
use slint::{ComponentHandle, Image, Rgb8Pixel, SharedPixelBuffer, Weak};

use crate::{
    training::{training_stats_recorder::TrainingStats, types::TrainingState}, AppWindow, PlotSize, Plots,
    UiTrainingStats,
};

use std::sync::{Arc, Mutex};
use std::{
    sync::mpsc::*,
    thread::{self, JoinHandle},
};

#[derive(Default)]
pub(crate) struct PlotsSizes {
    pub score_plot_size: PlotSize,
    pub epoch_legth_plot_size: PlotSize,
    pub reward_plot_size: PlotSize,
    pub best_tile_plot_size: PlotSize,
}

impl PlotsSizes {
    pub(crate) fn new(
        score_plot_size: PlotSize,
        epoch_legth_plot_size: PlotSize,
        reward_plot_size: PlotSize,
        best_tile_plot_size: PlotSize,
    ) -> PlotsSizes {
        Self {
            score_plot_size,
            epoch_legth_plot_size,
            reward_plot_size,
            best_tile_plot_size,
        }
    }

    pub(crate) fn from(area_size: PlotSize) -> PlotsSizes {
        let spacing = 8;
        let score = PlotSize {
            width: (area_size.width - spacing) / 2,
            height: (area_size.height - spacing) / 2,
        };
        let epoch_legth = PlotSize {
            width: (area_size.width - spacing) / 2,
            height: (area_size.height - spacing) / 2,
        };
        let reward = PlotSize {
            width: (area_size.width - spacing) / 2,
            height: (area_size.height - spacing) / 2,
        };
        let best_tile = PlotSize {
            width: (area_size.width - spacing) / 2,
            height: (area_size.height - spacing) / 2,
        };
        PlotsSizes {
            score_plot_size: score,
            epoch_legth_plot_size: epoch_legth,
            reward_plot_size: reward,
            best_tile_plot_size: best_tile,
        }
    }
}

impl PlotSize {
    pub(crate) fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }
}

pub(crate) enum PlotRangeType {
    All,
    LastEpochs(usize),
    Custom(usize, usize),
}

pub(crate) struct PlotsSettings {
    pub is_log_scale_enabled: bool,
    pub range: PlotRangeType,
}

impl Default for PlotsSettings {
    fn default() -> Self {
        Self {
            is_log_scale_enabled: false,
            range: PlotRangeType::All,
        }
    }
}

pub(crate) enum TrainingOverviewUpdate {
    EpochFinished(TrainingStats),
    StateChanged(TrainingState),
    PlotsSizesChanged(PlotsSizes),
    PlotsSettingsChanged(PlotsSettings),
}

pub(crate) struct TrainingOverviewThread {
    ui_handle: Weak<AppWindow>,
    scores: Vec<u32>,
    epoch_length: Vec<u32>,
    rewards: Vec<f32>,
    best_tiles: Vec<u32>,
    best_score: u32,
    best_tile: u32,
    plots_sizes: PlotsSizes,
    plots_settings: PlotsSettings,
    epoch_per_second_counter: Arc<Mutex<u32>>,
}

impl TrainingOverviewThread {
    pub(crate) fn spawn_thread(
        ui_handle: Weak<AppWindow>,
    ) -> (
        Sender<TrainingOverviewUpdate>,
        JoinHandle<()>,
        Arc<Mutex<u32>>,
    ) {
        let (tx, rx) = channel();
        let mut thread = Self::new(ui_handle);
        let epochs_per_second = thread.epoch_per_second_counter.clone();

        let handle = thread::spawn(move || {
            thread.execute(rx);
        });

        (tx, handle, epochs_per_second)
    }

    fn new(ui_handle: Weak<AppWindow>) -> Self {
        Self {
            ui_handle,
            scores: Vec::new(),
            epoch_length: Vec::new(),
            rewards: Vec::new(),
            best_tiles: Vec::new(),
            best_score: 0,
            best_tile: 0,
            plots_sizes: Default::default(),
            plots_settings: Default::default(),
            epoch_per_second_counter: Arc::new(Mutex::new(0)),
        }
    }

    fn execute(&mut self, updates: Receiver<TrainingOverviewUpdate>) {
        use TrainingOverviewUpdate::*;

        loop {
            for update in &updates {
                match update {
                    EpochFinished(stats) => {
                        self.handle_new_epoch_stats(stats);
                    }
                    StateChanged(state) => {
                        self.handle_new_state(state);
                    }
                    PlotsSizesChanged(sizes) => {
                        self.handle_plot_size_change(sizes);
                    }
                    PlotsSettingsChanged(settings) => {
                        self.handle_plots_settings_change(settings);
                    }
                }
            }
        }
    }

    fn handle_new_epoch_stats(&mut self, training_stats: TrainingStats) {
        self.scores.push(training_stats.last_epoch_score);
        self.epoch_length.push(training_stats.last_epoch_length);
        self.rewards.push(training_stats.cumulated_epoch_rewards);
        self.best_tiles.push(training_stats.best_tile);
        self.best_score = self.best_score.max(training_stats.last_epoch_score);
        self.best_tile = self.best_tile.max(training_stats.best_tile);

        let best_score = self.best_score;
        let best_tile = self.best_tile;
        let ui_handle = self.ui_handle.clone();

        slint::invoke_from_event_loop(move || {
            let ui_handle = ui_handle.clone();
            let ui = ui_handle.unwrap();
            let stats = ui.global::<UiTrainingStats>();

            stats.set_epoch(training_stats.epochs as i32);
            stats.set_best_score(best_score as i32);
            stats.set_best_tile(best_tile as i32);
            stats.set_recorded_states(training_stats.replay_buffer_size as i32);
            stats.set_epsilon(training_stats.epsilon as f32);
        })
        .unwrap();

        self.update_plots();

        let mut counter = self.epoch_per_second_counter.lock().unwrap();
        *counter += 1;
    }

    fn handle_new_state(&mut self, state: TrainingState) {
        let ui_handle = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
            let ui_handle = ui_handle.clone();
            let ui = ui_handle.unwrap();
            let stats = ui.global::<UiTrainingStats>();
            stats.set_state(state.as_ui_training_state());
        })
        .unwrap();
    }

    fn handle_plot_size_change(&mut self, plot_sizes: PlotsSizes) {
        let mut score_plot = None;
        let mut epoch_length_plot = None;
        let mut reward_plot = None;
        let mut best_tile_plot = None;

        if self.plots_sizes.score_plot_size != plot_sizes.score_plot_size
            && plot_sizes.score_plot_size.is_valid()
        {
            self.plots_sizes.score_plot_size = plot_sizes.score_plot_size;
            score_plot = Some(render_score_plot(
                "score per epoch",
                &self.scores,
                self.plots_sizes.score_plot_size.width as u32,
                self.plots_sizes.score_plot_size.height as u32,
                &self.plots_settings,
            ));
        }
        if self.plots_sizes.epoch_legth_plot_size != plot_sizes.epoch_legth_plot_size
            && plot_sizes.epoch_legth_plot_size.is_valid()
        {
            self.plots_sizes.epoch_legth_plot_size = plot_sizes.epoch_legth_plot_size;
            epoch_length_plot = Some(render_score_plot(
                "game length per epoch",
                &self.epoch_length,
                self.plots_sizes.epoch_legth_plot_size.width as u32,
                self.plots_sizes.epoch_legth_plot_size.height as u32,
                &self.plots_settings,
            ));
        }
        if self.plots_sizes.reward_plot_size != plot_sizes.reward_plot_size
            && plot_sizes.reward_plot_size.is_valid()
        {
            self.plots_sizes.reward_plot_size = plot_sizes.reward_plot_size;
            reward_plot = Some(render_reward_plot(
                "reward per epoch",
                &self.rewards,
                self.plots_sizes.reward_plot_size.width as u32,
                self.plots_sizes.reward_plot_size.height as u32,
                &self.plots_settings,
            ));
        }
        if self.plots_sizes.best_tile_plot_size != plot_sizes.best_tile_plot_size
            && plot_sizes.best_tile_plot_size.is_valid()
        {
            self.plots_sizes.best_tile_plot_size = plot_sizes.best_tile_plot_size;
            best_tile_plot = Some(render_best_tile_plot(
                "best tile per epoch",
                &self.best_tiles,
                self.plots_sizes.best_tile_plot_size.width as u32,
                self.plots_sizes.best_tile_plot_size.height as u32,
                &self.plots_settings,
            ));
        }

        let ui_handle = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
            let ui = ui_handle.unwrap();
            let plots = ui.global::<Plots>();

            if let Some(score_plot) = score_plot {
                plots.set_score_plot(Image::from_rgb8(score_plot));
            }
            if let Some(epoch_length_plot) = epoch_length_plot {
                plots.set_epoch_length_plot(Image::from_rgb8(epoch_length_plot));
            }
            if let Some(reward_plot) = reward_plot {
                plots.set_reward_plot(Image::from_rgb8(reward_plot));
            }
            if let Some(best_tile_plot) = best_tile_plot {
                plots.set_best_tile_plot(Image::from_rgb8(best_tile_plot));
            }
        })
        .unwrap();
    }

    fn handle_plots_settings_change(&mut self, settings: PlotsSettings) {
        self.plots_settings = settings;
        self.update_plots();
    }

    fn update_plots(&self) {
        let score_plot = render_score_plot(
            "score per epoch",
            &self.scores,
            self.plots_sizes.score_plot_size.width as u32,
            self.plots_sizes.score_plot_size.height as u32,
            &self.plots_settings,
        );
        let epoch_length_plot = render_score_plot(
            "game length per epoch",
            &self.epoch_length,
            self.plots_sizes.epoch_legth_plot_size.width as u32,
            self.plots_sizes.epoch_legth_plot_size.height as u32,
            &self.plots_settings,
        );
        let reward_plot = render_reward_plot(
            "reward per epoch",
            &self.rewards,
            self.plots_sizes.reward_plot_size.width as u32,
            self.plots_sizes.reward_plot_size.height as u32,
            &self.plots_settings,
        );
        let best_tile_plot = render_best_tile_plot(
            "best tile per epoch",
            &self.best_tiles,
            self.plots_sizes.best_tile_plot_size.width as u32,
            self.plots_sizes.best_tile_plot_size.height as u32,
            &self.plots_settings,
        );

        let ui_handle = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
            let ui = ui_handle.unwrap();
            let plots = ui.global::<Plots>();

            plots.set_score_plot(Image::from_rgb8(score_plot));
            plots.set_epoch_length_plot(Image::from_rgb8(epoch_length_plot));
            plots.set_reward_plot(Image::from_rgb8(reward_plot));
            plots.set_best_tile_plot(Image::from_rgb8(best_tile_plot));
        })
        .unwrap();
    }
}

fn render_score_plot(
    caption: &str,
    values: &[u32],
    width: u32,
    height: u32,
    settings: &PlotsSettings,
) -> SharedPixelBuffer<Rgb8Pixel> {
    let start_x: usize;
    let end_x: usize;
    let values = match settings.range {
        PlotRangeType::All => {
            start_x = 1;
            end_x = values.len().max(width as usize);
            values
        }
        PlotRangeType::LastEpochs(epochs) => {
            start_x = values.len().saturating_sub(epochs) + 1;
            end_x = start_x + epochs - 1;
            &values[values.len().saturating_sub(epochs)..]
        }
        PlotRangeType::Custom(start, end) => {
            let start = start.min(values.len().saturating_sub(1));
            let end = end.min(values.len().saturating_sub(1));
            start_x = start;
            end_x = end;
            if values.is_empty() {
                values
            } else {
                &values[start..=end]
            }
        }
    };
    let y_axis_length = if settings.is_log_scale_enabled {
        values.iter().max().unwrap_or(&1).ilog2()
    } else {
        *values.iter().max().unwrap_or(&0)
    };

    let mut pixel_buffer = SharedPixelBuffer::new(width, height);
    let backend = BitMapBackend::with_buffer(pixel_buffer.make_mut_bytes(), (width, height));

    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(8)
        .x_label_area_size(20)
        .y_label_area_size(30)
        .build_cartesian_2d(start_x..end_x, 0..y_axis_length)
        .expect("failed to build chart");

    if settings.is_log_scale_enabled {
        chart
            .configure_mesh()
            .y_label_formatter(&|y| format!("{}", 2_u32.pow(*y)))
            .draw()
            .unwrap();
        let points = (start_x..=end_x)
            .zip(values.iter())
            .map(|(x, y)| (x, y.ilog2()));
        chart.draw_series(LineSeries::new(points, RED)).unwrap();
    } else {
        chart.configure_mesh().draw().unwrap();
        let points = (start_x..=end_x).zip(values.iter()).map(|(x, y)| (x, *y));
        chart.draw_series(LineSeries::new(points, RED)).unwrap();
    }

    drop(chart);
    drop(root);

    pixel_buffer
}

fn render_reward_plot(
    caption: &str,
    values: &[f32],
    width: u32,
    height: u32,
    settings: &PlotsSettings,
) -> SharedPixelBuffer<Rgb8Pixel> {
    let start_x: usize;
    let end_x: usize;
    let values = match settings.range {
        PlotRangeType::All => {
            start_x = 1;
            end_x = values.len().max(width as usize);
            values
        }
        PlotRangeType::LastEpochs(epochs) => {
            start_x = values.len().saturating_sub(epochs) + 1;
            end_x = start_x + epochs - 1;
            &values[values.len().saturating_sub(epochs)..]
        }
        PlotRangeType::Custom(start, end) => {
            let start = start.min(values.len().saturating_sub(1));
            let end = end.min(values.len().saturating_sub(1));
            start_x = start;
            end_x = end;
            if values.is_empty() {
                values
            } else {
                &values[start..=end]
            }
        }
    };
    let y_axis_length = if settings.is_log_scale_enabled {
        values
            .iter()
            .max_by(|lhs, rhs| lhs.total_cmp(rhs))
            .unwrap_or(&0f32)
            .log2()
    } else {
        *values
            .iter()
            .max_by(|lhs, rhs| lhs.total_cmp(rhs))
            .unwrap_or(&0f32)
    };

    let mut pixel_buffer = SharedPixelBuffer::new(width, height);
    let backend = BitMapBackend::with_buffer(pixel_buffer.make_mut_bytes(), (width, height));

    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(8)
        .x_label_area_size(20)
        .y_label_area_size(30)
        .build_cartesian_2d(start_x..end_x, 0f32..y_axis_length)
        .expect("failed to build chart");

    if settings.is_log_scale_enabled {
        chart
            .configure_mesh()
            .y_label_formatter(&|y| format!("{}", 2_f32.powf(*y)))
            .draw()
            .unwrap();
        let points = (start_x..=end_x).zip(values).map(|(x, y)| (x, y.log2()));
        chart.draw_series(LineSeries::new(points, RED)).unwrap();
    } else {
        chart.configure_mesh().draw().unwrap();
        let points = (start_x..=end_x).zip(values).map(|(x, y)| (x, *y));
        chart.draw_series(LineSeries::new(points, RED)).unwrap();
    }

    drop(chart);
    drop(root);

    pixel_buffer
}

fn render_best_tile_plot(
    caption: &str,
    values: &[u32],
    width: u32,
    height: u32,
    settings: &PlotsSettings,
) -> SharedPixelBuffer<Rgb8Pixel> {
    let start_x: usize;
    let end_x: usize;
    let values = match settings.range {
        PlotRangeType::All => {
            start_x = 1;
            end_x = values.len().max(width as usize);
            values
        }
        PlotRangeType::LastEpochs(epochs) => {
            start_x = values.len().saturating_sub(epochs) + 1;
            end_x = start_x + epochs - 1;
            &values[values.len().saturating_sub(epochs)..]
        }
        PlotRangeType::Custom(start, end) => {
            let start = start.min(values.len().saturating_sub(1));
            let end = end.min(values.len().saturating_sub(1));
            start_x = start;
            end_x = end;
            if values.is_empty() {
                values
            } else {
                &values[start..=end]
            }
        }
    };

    let mut pixel_buffer = SharedPixelBuffer::new(width, height);
    let backend = BitMapBackend::with_buffer(pixel_buffer.make_mut_bytes(), (width, height));

    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(8)
        .x_label_area_size(20)
        .y_label_area_size(30)
        .build_cartesian_2d(
            start_x..end_x,
            0..(values.iter().max().unwrap_or(&1).ilog2() + 1),
        )
        .expect("failed to build chart");

    chart
        .configure_mesh()
        .y_label_formatter(&|y| format!("{}", 2_i32.pow(*y)))
        .draw()
        .unwrap();

    let points = (start_x..=end_x).zip(values).map(|(x, y)| (x, y.ilog2()));

    chart.draw_series(LineSeries::new(points, RED)).unwrap();

    drop(chart);
    drop(root);

    pixel_buffer
}
