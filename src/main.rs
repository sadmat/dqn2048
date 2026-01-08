// Prevent console window in addition to Slint window in Windows release builds when, e.g., starting the app via file manager. Ignored on other platforms.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod dqn;
mod game;
mod training;
mod ui;

use crate::training::training_thread::TrainingThread;
use crate::training::types::TrainingAction;
use crate::ui::training_overview::TrainingOverviewUpdate::PlotsSizesChanged;
use crate::ui::training_overview::{
    PlotRangeType, PlotsSettings, PlotsSizes, TrainingOverviewThread, TrainingOverviewUpdate,
};
use crate::ui::training_update_adapter::TrainingUpdateAdapter;
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use num_format::{SystemLocale, ToFormattedString};
use plotters::prelude::*;
use rfd::FileDialog;
use slint::{Timer, TimerMode, Weak, quit_event_loop};
use std::error::Error;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

slint::include_modules!();

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "rocm")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Rocm>>::spawn_thread();
    #[cfg(feature = "cuda")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Cuda>>::spawn_thread();
    #[cfg(feature = "wgpu")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Wgpu>>::spawn_thread();

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    let (updates_tx, _, epochs_per_second) =
        TrainingOverviewThread::spawn_thread(ui_handle.clone());
    let _ = TrainingUpdateAdapter::spawn_thread(messages_rx, updates_tx.clone());

    setup_actions(actions_tx, &ui, updates_tx.clone());
    setup_plots(&ui, updates_tx);
    start_plots_area_update_timer(ui_handle.clone());
    let _timer = start_epochs_per_second_timer(epochs_per_second, ui_handle.clone());
    setup_formatters(ui_handle.clone());

    ui.run()?;

    Ok(())
}

fn setup_actions(
    actions_tx: Sender<TrainingAction>,
    ui: &AppWindow,
    updates_tx: Sender<TrainingOverviewUpdate>,
) {
    let actions = ui.global::<Actions>();
    let ui_handle = ui.as_weak();

    actions.on_start_training({
        let action_tx = actions_tx.clone();
        move || {
            action_tx.send(TrainingAction::Start).unwrap();
        }
    });
    actions.on_pause_training({
        let action_tx = actions_tx.clone();
        move || {
            action_tx.send(TrainingAction::Pause).unwrap();
        }
    });
    actions.on_save_model({
        let action_tx = actions_tx.clone();
        move || {
            let Some(file) = FileDialog::new().save_file() else {
                return;
            };
            action_tx.send(TrainingAction::SaveModel(file)).unwrap();
        }
    });
    actions.on_load_model({
        let action_tx = actions_tx.clone();
        move || {
            let Some(file) = FileDialog::new().pick_file() else {
                return;
            };
            action_tx.send(TrainingAction::LoadModel(file)).unwrap();
        }
    });
    actions.on_load_session({
        let action_tx = actions_tx.clone();
        move || {
            let Some(dir) = FileDialog::new().pick_folder() else {
                return;
            };
            action_tx.send(TrainingAction::LoadSession(dir)).unwrap();
        }
    });
    actions.on_save_session({
        let action_tx = actions_tx.clone();
        move || {
            let Some(dir) = FileDialog::new().pick_folder() else {
                return;
            };
            action_tx.send(TrainingAction::SaveSession(dir)).unwrap();
        }
    });
    actions.on_quit(|| {
        quit_event_loop().unwrap();
    });
}

fn setup_plots(ui: &AppWindow, updates_tx: Sender<TrainingOverviewUpdate>) {
    let plots = ui.global::<Plots>();
    let ui_handle = ui.as_weak();
    plots.on_plots_area_size_changed({
        let ui_handle = ui_handle.clone();
        let updates_tx = updates_tx.clone();
        move || {
            let ui = ui_handle.unwrap();
            let plots = ui.global::<Plots>();
            let area_size = plots.get_plots_area_size();

            let sizes = PlotsSizes::from(area_size);
            updates_tx.send(PlotsSizesChanged(sizes)).unwrap();
        }
    });
    plots.on_range_settings_changed({
        move || {
            let ui = ui_handle.unwrap();
            let plots = ui.global::<Plots>();
            if let Some(settings) = build_plot_settings(&plots) {
                updates_tx
                    .send(TrainingOverviewUpdate::PlotsSettingsChanged(settings))
                    .unwrap();
            }
        }
    });
}

fn build_plot_settings(plots: &Plots) -> Option<PlotsSettings> {
    let range_settings = match plots.get_range_type() {
        UiPlotRangeType::All => Some(PlotRangeType::All),
        UiPlotRangeType::LastEpochs => {
            Some(PlotRangeType::LastEpochs(plots.get_last_epochs() as usize))
        }
        UiPlotRangeType::Custom => {
            let range_start = plots.get_custom_range_start() as usize;
            let range_end = plots.get_custom_range_end() as usize;
            if range_start < range_end {
                Some(PlotRangeType::Custom(range_start, range_end))
            } else {
                None
            }
        }
    };

    if let Some(range) = range_settings {
        return Some(PlotsSettings {
            is_log_scale_enabled: plots.get_log_scale(),
            range,
        });
    } else {
        None
    }
}

fn start_plots_area_update_timer(ui_handle: Weak<AppWindow>) {
    Timer::single_shot(Duration::from_millis(10), move || {
        let ui = ui_handle.unwrap();
        ui.invoke_force_plots_area_size_update();
    });
}

fn start_epochs_per_second_timer(
    epochs_per_second: Arc<Mutex<u32>>,
    ui_handle: Weak<AppWindow>,
) -> Timer {
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_secs(1), move || {
        let mut counter = epochs_per_second.lock().unwrap();
        let ui = ui_handle.unwrap();
        let stats = ui.global::<UiTrainingStats>();
        stats.set_epochs_per_second(*counter as i32);
        *counter = 0;
    });
    timer
}

fn setup_formatters(ui_handle: Weak<AppWindow>) {
    let ui = ui_handle.unwrap();
    let formatters = ui.global::<Formatters>();
    formatters.on_format_int(|value| {
        let locale = SystemLocale::default().unwrap();
        value.to_formatted_string(&locale).into()
    });
}
