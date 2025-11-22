// Prevent console window in addition to Slint window in Windows release builds when, e.g., starting the app via file manager. Ignored on other platforms.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod dqn;
mod game;
mod training;

use crate::training::training_thread::TrainingThread;
use crate::training::types::{TrainingAction, TrainingMessage};
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
use slint::quit_event_loop;
use std::error::Error;
use std::thread;

slint::include_modules!();

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "rocm")]
    let (actions, messages, handle) = TrainingThread::<Autodiff<Rocm>>::spawn_thread();
    #[cfg(feature = "cuda")]
    let (actions_tx, messages_rx, handle) = TrainingThread::<Autodiff<Cuda>>::spawn_thread();

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    let actions = ui.global::<Actions>();

    actions.on_start_training({
        let action_tx = actions_tx.clone();
        move || { action_tx.send(TrainingAction::Start).unwrap(); }
    });
    actions.on_pause_training({
        let action_tx = actions_tx.clone();
        move || { action_tx.send(TrainingAction::Pause).unwrap(); }
    });
    actions.on_save_model(|| {
        println!("TODO: on_save_model()");
    });
    actions.on_load_model(|| {
        println!("TODO: on_load_model()");
    });
    actions.on_plot_size_changed(|| {
        println!("TODO: on_plot_size_changed()");
    });
    actions.on_quit(|| {
        quit_event_loop().unwrap();
    });

    thread::spawn(move || {
        let mut best_score: u32 = 0;
        let mut best_tile: u32 = 0;

        for message in messages_rx {
            let ui_handle = ui_handle.clone();

            match message {
                TrainingMessage::StateChanged(state) => {
                    slint::invoke_from_event_loop(move || {
                        let ui_handle = ui_handle.clone();
                        let ui = ui_handle.unwrap();
                        let stats = ui.global::<TrainingStats>();
                        stats.set_state(state.as_ui_training_state());
                    }).unwrap();
                },
                TrainingMessage::EpochFinished(epoch_stats) => {
                    best_score = best_score.max(epoch_stats.last_epoch_score);
                    best_tile = best_tile.max(epoch_stats.best_tile);

                    slint::invoke_from_event_loop(move || {
                        let ui_handle = ui_handle.clone();
                        let ui = ui_handle.unwrap();
                        let stats = ui.global::<TrainingStats>();
                        stats.set_epoch(epoch_stats.epochs as i32);
                        stats.set_epochs_per_second(epoch_stats.epochs_per_second.unwrap_or(0.0));
                        stats.set_best_score(best_score as i32);
                        stats.set_best_tile(best_tile as i32);
                    }).unwrap();
                },
            }
        }
    });

    ui.run()?;

    Ok(())
}

impl crate::training::types::TrainingState {
    fn as_ui_training_state(&self) -> UiTrainingState {
        match self {
            crate::training::types::TrainingState::Idle => UiTrainingState::Idle,
            crate::training::types::TrainingState::Training => UiTrainingState::Training,
        }
    }
}
