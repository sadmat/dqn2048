// Prevent console window in addition to Slint window in Windows release builds when, e.g., starting the app via file manager. Ignored on other platforms.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod dqn;
mod game;
mod training;

use crate::training::training_thread::TrainingThread;
use crate::training::types::TrainingAction;
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
use slint::quit_event_loop;
use std::error::Error;

slint::include_modules!();

fn main() -> Result<(), Box<dyn Error>> {
    let ui = AppWindow::new()?;

    let actions = ui.global::<Actions>();

    actions.on_start_training({
        let ui_handle = ui.as_weak();
        move || {
            let ui = ui_handle.unwrap();
            let stats = ui.global::<TrainingStats>();
            stats.set_state(TrainingState::Training);
            println!("TODO: on_start_training()");
        }
    });
    actions.on_pause_training({
        let ui_handle = ui.as_weak();
        move || {
            let ui = ui_handle.unwrap();
            let stats = ui.global::<TrainingStats>();
            stats.set_state(TrainingState::Idle);
            println!("TODO: on_start_training()");
        }
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

    #[cfg(feature = "rocm")]
    let (actions, messages, handle) = TrainingThread::<Autodiff<Rocm>>::spawn_thread();
    #[cfg(feature = "cuda")]
    let (actions, messages, handle) = TrainingThread::<Autodiff<Cuda>>::spawn_thread();

    actions.send(TrainingAction::Start).unwrap();
    println!("Training thread started");
    // for message in messages {
    //     println!("{:?}", message);
    // }

    ui.run()?;

    Ok(())
}
