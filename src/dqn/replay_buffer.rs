use std::collections::VecDeque;
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use rand::seq::IndexedRandom;

use crate::dqn::{state::StateType, training_batch::TrainingBatch};
use crate::dqn::data_augmenter::DataAugmenterType;

pub struct ReplayBuffer<S: StateType, D: DataAugmenterType<State = S>> {
    data_augmenter: D,
    transitions: VecDeque<StateTransition<S>>,
    capacity: usize,
}

impl<S: StateType, D: DataAugmenterType<State = S>> ReplayBuffer<S, D> {
    pub fn new(data_augmenter: D, capacity: usize) -> Self {
        ReplayBuffer {
            data_augmenter,
            transitions: VecDeque::new(),
            capacity,
        }
    }

    pub fn store(&mut self, transition: StateTransition<S>) {
        let new_transitions = self.data_augmenter.augment(transition);

        while self.transitions.len() + new_transitions.len() > self.capacity {
            self.transitions.pop_front();
        }

        self.transitions.extend(new_transitions);
    }

    pub fn size(&self) -> usize {
        self.transitions.len()
    }

    pub fn sample<B: AutodiffBackend>(&self, batch_size: usize) -> TrainingBatch<B> {
        let mut rng = rand::rng();
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let index = rng.random_range(0..self.transitions.len());
            batch.push(&self.transitions[index]);
        }
        batch.into()
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
