use std::{
    sync::mpsc::{Receiver, Sender},
    thread::{self, JoinHandle},
};

use crate::{training::types::TrainingMessage, ui::training_overview::TrainingOverviewUpdate};

pub(crate) struct TrainingUpdateAdapter {}

impl TrainingUpdateAdapter {
    pub(crate) fn spawn_thread(
        messages_rx: Receiver<TrainingMessage>,
        update_tx: Sender<TrainingOverviewUpdate>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            for message in messages_rx {
                match message {
                    TrainingMessage::StateChanged(state) => {
                        update_tx
                            .send(TrainingOverviewUpdate::StateChanged(state))
                            .unwrap();
                    }
                    TrainingMessage::EpochFinished(stats) => {
                        update_tx.send(TrainingOverviewUpdate::EpochFinished(stats));
                    }
                }
            }
        })
    }
}
