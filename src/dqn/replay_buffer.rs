use burn::prelude::Backend;
use rand::seq::IndexedRandom;

use crate::dqn::{state::State, training_batch::TrainingBatch};

pub struct ReplayBuffer<const N: usize, S: State<N>> {
    transitions: Vec<StateTransition<N, S>>,
}

impl<const N: usize, S: State<N>> ReplayBuffer<N, S> {
    pub fn new() -> Self {
        ReplayBuffer {
            transitions: Vec::new(),
        }
    }

    pub fn store(&mut self, transition: StateTransition<N, S>) {
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

pub struct StateTransition<const N: usize, S: State<N>> {
    pub state: S,
    pub action: S::Action,
    pub reward: f32,
    pub next_state: S,
}

impl<const N: usize, S: State<N>> StateTransition<N, S> {
    pub fn new(state: S, action: S::Action, reward: f32, next_state: S) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
        }
    }
}
