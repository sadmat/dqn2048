// Prevent console window in addition to Slint window in Windows release builds when, e.g., starting the app via file manager. Ignored on other platforms.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod dqn;
mod game;
mod training;

use std::error::Error;
use burn::backend::{Autodiff, Rocm};
use slint::quit_event_loop;
use crate::training::training_thread::TrainingThread;
use crate::training::types::TrainingAction;

slint::include_modules!();

fn main() -> Result<(), Box<dyn Error>> {
    let ui = AppWindow::new()?;

    ui.on_plot_size_changed(|size| {
        println!("OMG! It works...?!!! {:?}", size);
    });

    ui.on_quit(|| {
        quit_event_loop().unwrap();
    });

    // ui.on_request_increase_value({
    //     let ui_handle = ui.as_weak();
    //     move || {
    //         let ui = ui_handle.unwrap();
    //         ui.set_counter(ui.get_counter() + 1);
    //     }
    // });

    let (actions, messages, handle) = TrainingThread::<Autodiff<Rocm>>::spawn_thread();
    actions.send(TrainingAction::Start).unwrap();
    println!("Training thread started");
    for message in messages {
        println!("{:?}", message);
    }

    ui.run()?;

    Ok(())
}
