use burn::prelude::Backend;
use rand::seq::IndexedRandom;

use crate::dqn::{state::StateType, training_batch::TrainingBatch};

pub struct ReplayBuffer<S: StateType> {
    transitions: Vec<StateTransition<S>>,
}

impl<S: StateType> ReplayBuffer<S> {
    pub fn new() -> Self {
        ReplayBuffer {
            transitions: Vec::new(),
        }
    }

    pub fn store(&mut self, transition: StateTransition<S>) {
        self.transitions.push(transition);
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
