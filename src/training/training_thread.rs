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
    training::types::{TrainingAction, TrainingMessage},
};

struct TrainingThread<B: AutodiffBackend> {
    actions: Receiver<TrainingAction>,
    messages: Sender<TrainingMessage>,
    trainer: Trainer<B, GameModel, Board<RealGameRng>, TrainingCritic>,
}

struct TrainingCritic {}

impl CriticType for TrainingCritic {
    type State = Board<RealGameRng>;

    fn reward(
        &self,
        state: &Self::State,
        action: &<Self::State as crate::dqn::state::StateType>::Action,
        next_state: &Self::State,
    ) -> f32 {
        todo!()
    }
}

#[derive(Module, Clone, Debug)]
struct GameModel {}

impl<B: Backend> Model<B> for GameModel {
    fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        todo!()
    }
}
