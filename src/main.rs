// Prevent console window in addition to Slint window in Windows release builds when, e.g., starting the app via file manager. Ignored on other platforms.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod dqn;
mod game;
mod training;
mod ui;

use crate::training::training_thread::TrainingThread;
use crate::training::types::{TrainingAction, TrainingMessage};
use crate::ui::training_overview::TrainingOverviewUpdate::PlotsSizesChanged;
use crate::ui::training_overview::{PlotsSizes, TrainingOverviewThread};
use crate::ui::training_update_adapter::TrainingUpdateAdapter;
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
use plotters::prelude::*;
use slint::{Image, Rgb8Pixel, SharedPixelBuffer, quit_event_loop};
use std::error::Error;
use std::thread;

slint::include_modules!();

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "rocm")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Rocm>>::spawn_thread();
    #[cfg(feature = "cuda")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Cuda>>::spawn_thread();

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    let (updates_tx, _) = TrainingOverviewThread::spawn_thread(ui_handle.clone());
    let _ = TrainingUpdateAdapter::spawn_thread(messages_rx, updates_tx.clone());

    let actions = ui.global::<Actions>();

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
    actions.on_save_model(|| {
        println!("TODO: on_save_model()");
    });
    actions.on_load_model(|| {
        println!("TODO: on_load_model()");
    });
    actions.on_plot_size_changed(move || {
        let ui = ui_handle.unwrap();
        let plots = ui.global::<Plots>();
        let sizes = PlotsSizes::new(
            plots.get_score_plot_size(),
            plots.get_reward_plot_size(),
            plots.get_best_tile_plot_size(),
        );
        updates_tx.send(PlotsSizesChanged(sizes)).unwrap();
    });
    actions.on_quit(|| {
        quit_event_loop().unwrap();
    });

    ui.run()?;

    Ok(())
}
