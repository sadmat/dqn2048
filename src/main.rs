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
use slint::{Image, Rgb8Pixel, SharedPixelBuffer, Timer, quit_event_loop, Weak, TimerMode};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

slint::include_modules!();

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "rocm")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Rocm>>::spawn_thread();
    #[cfg(feature = "cuda")]
    let (actions_tx, messages_rx, _) = TrainingThread::<Autodiff<Cuda>>::spawn_thread();

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    let (updates_tx, _, epochs_per_second) = TrainingOverviewThread::spawn_thread(ui_handle.clone());
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
    actions.on_plots_area_size_changed({
        let ui_handle = ui_handle.clone();
        move || {
            let ui = ui_handle.unwrap();
            let plots = ui.global::<Plots>();
            let area_size = plots.get_plots_area_size();

            let sizes = PlotsSizes::from(area_size);
            updates_tx.send(PlotsSizesChanged(sizes)).unwrap();
        }
    });
    actions.on_quit(|| {
        quit_event_loop().unwrap();
    });

    start_plots_area_update_timer(ui_handle.clone());
    let _timer = start_epochs_per_second_timer(epochs_per_second, ui_handle.clone());

    ui.run()?;

    Ok(())
}

fn start_plots_area_update_timer(ui_handle: Weak<AppWindow>) {
    Timer::single_shot(Duration::from_millis(10), move || {
        let ui = ui_handle.unwrap();
        ui.invoke_force_plots_area_size_update();
    });
}

fn start_epochs_per_second_timer(epochs_per_second: Arc<Mutex<u32>>, ui_handle: Weak<AppWindow>) -> Timer {
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