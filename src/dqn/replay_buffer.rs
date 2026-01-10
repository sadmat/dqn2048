use burn::tensor::backend::AutodiffBackend;
use rand::seq::IndexedRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::dqn::data_augmenter::DataAugmenterType;
use crate::dqn::serialization::training_config::ReplayBufferConfig;
use crate::dqn::state::ActionType;
use crate::dqn::{state::StateType, training_batch::TrainingBatch};

pub struct ReplayBuffer<S: StateType, D: DataAugmenterType<State = S>> {
    data_augmenter: D,
    transitions: Vec<StateTransition>,
    capacity: usize,
    write_position: usize,
}

impl<S: StateType, D: DataAugmenterType<State = S>> ReplayBuffer<S, D> {
    pub fn new(data_augmenter: D, capacity: usize) -> Self {
        ReplayBuffer {
            data_augmenter,
            transitions: Vec::with_capacity(capacity),
            capacity,
            write_position: 0,
        }
    }

    pub fn from(transitions: Vec<StateTransition>, config: &ReplayBufferConfig) -> Self {
        ReplayBuffer {
            data_augmenter: Default::default(),
            transitions,
            capacity: config.capacity,
            write_position: config.write_position,
        }
    }

    pub fn store(&mut self, state: S, action: S::Action, reward: f32, next_state: S) {
        let new_transitions = self
            .data_augmenter
            .augment(state, action, reward, next_state);

        for transition in new_transitions {
            if self.transitions.len() < self.capacity {
                self.transitions.push(transition);
            } else {
                self.transitions[self.write_position] = transition;
            }
            self.write_position = (self.write_position + 1) % self.capacity;
        }
    }

    pub fn size(&self) -> usize {
        self.transitions.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn write_position(&self) -> usize {
        self.write_position
    }

    pub fn transitions(&self, from: usize, to: usize) -> &[StateTransition] {
        let start = from.min(self.transitions.len());
        let end = to.min(self.transitions.len());
        &self.transitions[start..end]
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

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct StateTransition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub invalid_actions_mask: Vec<bool>,
    pub is_terminal: f32,
}

impl StateTransition {
    pub fn new<S: StateType>(state: S, action: S::Action, reward: f32, next_state: S) -> Self {
        let mut invalid_actions = vec![true; S::NUM_ACTIONS];
        for valid_action in next_state.possible_actions() {
            invalid_actions[valid_action.index() as usize] = false;
        }

        Self {
            state: state.as_features(),
            action: action.index(),
            reward,
            next_state: next_state.as_features(),
            invalid_actions_mask: invalid_actions,
            is_terminal: if next_state.is_terminal() { 1.0 } else { 0.0 },
        }
    }
}
