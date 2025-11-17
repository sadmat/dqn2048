use std::sync::mpsc::{Receiver, Sender};

use burn::{
    Tensor,
    module::Module,
    prelude::{Backend, Float},
    tensor::backend::AutodiffBackend,
};

use crate::{
    dqn::{critic::CriticType, model::Model, trainer::Trainer},
    game::{board::Board, game_rng::RealGameRng},
    training::{
        game_model::GameModel,
        training_critic::TrainingCritic,
        types::{TrainingAction, TrainingMessage},
    },
};

struct TrainingThread<B: AutodiffBackend> {
    actions: Receiver<TrainingAction>,
    messages: Sender<TrainingMessage>,
    trainer: Trainer<B, GameModel<B>, Board<RealGameRng>, TrainingCritic>,
}
