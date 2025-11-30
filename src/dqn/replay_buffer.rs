use burn::prelude::Backend;
use rand::seq::IndexedRandom;

use crate::dqn::{state::StateType, training_batch::TrainingBatch};
use crate::dqn::data_augmenter::DataAugmenterType;

pub struct ReplayBuffer<S: StateType, D: DataAugmenterType<State = S>> {
    data_augmenter: D,
    transitions: Vec<StateTransition<S>>,
}

impl<S: StateType, D: DataAugmenterType<State = S>> ReplayBuffer<S, D> {
    pub fn new(data_augmenter: D) -> Self {
        ReplayBuffer {
            data_augmenter,
            transitions: Vec::new(),
        }
    }

    pub fn store(&mut self, transition: StateTransition<S>) {
        self.transitions.extend(self.data_augmenter.augment(transition));
    }

    pub fn size(&self) -> usize {
        self.transitions.len()
    }

    pub fn sample<B: Backend>(&self, batch_size: usize) -> TrainingBatch<B> {
        self.transitions
            .choose_multiple(&mut rand::rng(), batch_size)
            .collect::<Vec<_>>()
            .into()
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct StateTransition<S: StateType> {
    pub state: S,
    pub action: S::Action,
    pub reward: f32,
    pub next_state: S,
}

impl<S: StateType> StateTransition<S> {
    pub fn new(state: S, action: S::Action, reward: f32, next_state: S) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
        }
    }
}
