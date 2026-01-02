use burn::tensor::backend::AutodiffBackend;
use rand::seq::IndexedRandom;
use rand::Rng;

use crate::dqn::data_augmenter::DataAugmenterType;
use crate::dqn::state::ActionType;
use crate::dqn::sum_tree::SumTree;
use crate::dqn::{state::StateType, training_batch::TrainingBatch};

pub struct ReplayBuffer<S: StateType, D: DataAugmenterType<State = S>> {
    data_augmenter: D,
    transitions: Vec<StateTransition>,
    capacity: usize,
    write_position: usize,
    priorities: SumTree,
    max_priority: f32,
}

impl<S: StateType, D: DataAugmenterType<State = S>> ReplayBuffer<S, D> {
    pub fn new(data_augmenter: D, capacity: usize) -> Self {
        ReplayBuffer {
            data_augmenter,
            transitions: Vec::with_capacity(capacity),
            capacity,
            write_position: 0,
            priorities: SumTree::with_capacity(capacity),
            max_priority: 1.0,
        }
    }

    pub fn store(&mut self, state: S, action: S::Action, reward: f32, next_state: S) {
        let new_transitions = self
            .data_augmenter
            .augment(state, action, reward, next_state);

        for transition in new_transitions.into_iter() {
            if self.transitions.len() < self.capacity {
                self.transitions.push(transition);
            } else {
                self.transitions[self.write_position] = transition;
            }
            self.priorities
                .update(self.write_position, self.max_priority);
            self.write_position = (self.write_position + 1) % self.capacity;
        }
    }

    pub fn size(&self) -> usize {
        self.transitions.len()
    }

    pub fn sample<B: AutodiffBackend>(&self, batch_size: usize, beta: f32) -> TrainingBatch<B> {
        let mut rng = rand::rng();
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        // Sample indices per segment & prepare weights
        let total_priority = self.priorities.total();
        let segment_len = total_priority / batch_size as f32;
        for i in 0..batch_size {
            let segment_start = segment_len * i as f32;
            let segment_end = segment_start + segment_len;
            let value = rng
                .random_range(segment_start..segment_end)
                .min(total_priority - 0.0001);
            let (index, priority) = self.priorities.sample(value);
            let sample_prob = priority / total_priority;
            let weight = (sample_prob * self.transitions.len() as f32).powf(-beta);
            weights.push(weight);
            indices.push(index);
        }

        // Normalize weights
        let max_weight = weights.iter().map(|&weight| weight).fold(0.0, f32::max);
        for weight in &mut weights {
            *weight /= max_weight;
        }

        // Extract examples
        let examples = indices
            .iter()
            .map(|&index| &self.transitions[index])
            .collect::<Vec<_>>();

        TrainingBatch::from(examples, weights, indices)
    }

    pub fn update_priorities(
        &mut self,
        indices: &[usize],
        priorities: &[f32],
        alpha: f32,
        epsilon: f32,
    ) {
        for (index, priority) in indices.iter().zip(priorities) {
            let priority = (*priority + epsilon).powf(alpha);
            self.priorities.update(*index, priority);
            self.max_priority = f32::max(self.max_priority, priority);
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
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
