use crate::dqn::serialization::serializer::TrainingSerializer;
use crate::training::game_model::GameModelConfig;
use crate::training::training_data_augmenter::TrainingDataAugmenter;
use crate::training::training_stats_recorder::{TrainingStats, TrainingStatsRecorder};
use crate::training::types::TrainingState::Training;
use crate::{
    dqn::{
        critic::CriticType,
        model::Model,
        trainer::{Hyperparameters, Trainer},
    },
    game::{board::Board, game_rng::RealGameRng},
    training::{
        game_model::GameModel,
        training_critic::TrainingCritic,
        types::{TrainingAction, TrainingMessage, TrainingState},
    },
};
use burn::prelude::Device;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings, Recorder};
use burn::{
    Tensor,
    module::Module,
    prelude::{Backend, Float},
    tensor::backend::AutodiffBackend,
};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

pub(crate) struct TrainingThread<B: AutodiffBackend> {
    actions: Receiver<TrainingAction>,
    messages: Sender<TrainingMessage>,
    trainer: Trainer<
        B,
        GameModel<B>,
        Board<RealGameRng>,
        TrainingCritic,
        TrainingStatsRecorder,
        TrainingDataAugmenter,
    >,
    training_state: TrainingState,
}

impl<B: AutodiffBackend> TrainingThread<B> {
    pub(crate) fn spawn_thread() -> (
        Sender<TrainingAction>,
        Receiver<TrainingMessage>,
        JoinHandle<()>,
    ) {
        let (action_tx, action_rx) = mpsc::channel();
        let (message_tx, message_rx) = mpsc::channel();
        let mut thread = TrainingThread::<B>::new(action_rx, message_tx);

        let handle = thread::spawn(move || {
            thread.execute();
        });

        (action_tx, message_rx, handle)
    }

    fn new(
        actions: Receiver<TrainingAction>,
        messages: Sender<TrainingMessage>,
    ) -> TrainingThread<B> {
        let hyperparams = Hyperparameters::new();

        TrainingThread {
            actions,
            messages,
            trainer: Trainer::new(
                hyperparams,
                TrainingCritic::new(),
                TrainingDataAugmenter::default(),
                Default::default(),
            ),
            training_state: TrainingState::Idle,
        }
    }

    fn execute(&mut self) {
        let mut model = GameModelConfig::new().init(&Default::default());

        loop {
            model = self.handle_action(model);
            if self.training_state == TrainingState::Training {
                let (updated_model, stats) = self.trainer.run_epoch(model);
                model = updated_model;
                self.report_progress(stats);
            } else {
                thread::sleep(Duration::from_millis(200));
            }
        }
    }

    fn handle_action(&mut self, model: GameModel<B>) -> GameModel<B> {
        match self.actions.try_recv() {
            Ok(action) => {
                println!("Training thread received action {:?}", action);
                match action {
                    TrainingAction::Pause => {
                        self.training_state = TrainingState::Idle;
                        self.messages
                            .send(TrainingMessage::StateChanged(TrainingState::Idle))
                            .unwrap();
                    }
                    TrainingAction::Start => {
                        self.training_state = TrainingState::Training;
                        self.messages
                            .send(TrainingMessage::StateChanged(TrainingState::Training))
                            .unwrap();
                    }
                    TrainingAction::SaveModel(file_path) => {
                        self.save_model(&model, file_path);
                    }
                    TrainingAction::LoadModel(file_path) => {
                        return self.load_model(model, file_path);
                    }
                    TrainingAction::SaveSession(path) => {
                        self.save_session(&model, path);
                    }
                    TrainingAction::LoadSession(path) => {
                        self.load_session(path);
                    }
                }
            }
            Err(TryRecvError::Empty) => (),
            Err(TryRecvError::Disconnected) => panic!("Training thread disconnected"),
        }

        model
    }

    fn report_progress(&self, stats: TrainingStats) {
        self.messages
            .send(TrainingMessage::EpochFinished(stats))
            .unwrap();
    }

    fn save_model(&self, model: &GameModel<B>, file_path: PathBuf) {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();

        model
            .clone()
            .save_file(file_path, &recorder)
            .expect("Failed to save the model");
    }

    fn load_model(&self, model: GameModel<B>, file_path: PathBuf) -> GameModel<B> {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();

        model
            .load_file(file_path, &recorder, &Default::default())
            .expect("Failed to load the model")
    }

    fn save_session(&self, model: &GameModel<B>, path: PathBuf) {
        // TODO:
        // [x] Save model
        // [x] Save replay buffer
        // [x] Save hyperparameters and other session info
        // [ ] Save current plots
        // [ ] Report progress
        // [ ] Report errors
        match TrainingSerializer::serialize(&self.trainer, model.clone(), path) {
            Ok(()) => {
                println!("[dbg] serialization ok!");
            }
            Err(error) => {
                println!("[dbg] serialization failed: {}", error);
            }
        }
    }

    fn load_session(&mut self, path: PathBuf) {
        // TODO:
        // [ ] Purge previous training data
        // [ ] Load model
        // [ ] Load replay buffer
        // [ ] Load hyperparameters and other session info
        // [ ] Load current plots
    }
}
