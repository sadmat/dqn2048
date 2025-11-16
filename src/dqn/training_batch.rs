use crate::dqn::state::ActionType;
use crate::dqn::{replay_buffer::StateTransition, state::StateType};
use burn::prelude::TensorData;
use burn::{
    Tensor,
    prelude::Backend,
    tensor::{Bool, Float, Int},
};

pub struct TrainingBatch<B: Backend> {
    pub states: Tensor<B, 2, Float>,
    pub actions: Tensor<B, 1, Int>,
    pub invalid_actions_mask: Tensor<B, 2, Bool>,
    pub rewards: Tensor<B, 1, Float>,
    pub next_states: Tensor<B, 2, Float>,
    pub is_terminal: Tensor<B, 1, Float>,
}

impl<B: Backend, S: StateType> From<Vec<&StateTransition<S>>> for TrainingBatch<B> {
    fn from(records: Vec<&StateTransition<S>>) -> Self {
        let batch_size = records.len();
        let mut states: Vec<f32> = Vec::with_capacity(batch_size * S::num_features());
        let mut actions: Vec<i32> = Vec::with_capacity(batch_size);
        let mut invalid_actions_mask: Vec<bool> = Vec::with_capacity(batch_size * S::num_actions());
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut next_states: Vec<f32> = Vec::with_capacity(batch_size * S::num_features());
        let mut is_terminal: Vec<f32> = Vec::with_capacity(batch_size);

        for record in records {
            states.extend_from_slice(record.state.as_features().as_slice());
            actions.push(record.action.index() as i32);
            let mut invalid_actions = vec![true; S::num_actions()];
            for valid_action in record.next_state.possible_actions() {
                invalid_actions[valid_action.index() as usize] = false;
            }
            invalid_actions_mask.extend_from_slice(invalid_actions.as_slice());
            rewards.push(record.reward);
            next_states.extend_from_slice(record.next_state.as_features().as_slice());
            is_terminal.push(if record.next_state.is_terminal() {
                1.0
            } else {
                0.0
            });
        }

        let states = TensorData::new(states, [batch_size, S::num_features()]);
        let actions = TensorData::new(actions, [batch_size]);
        let invalid_actions_mask =
            TensorData::new(invalid_actions_mask, [batch_size, S::num_actions()]);
        let rewards = TensorData::new(rewards, [batch_size]);
        let next_states = TensorData::new(next_states, [batch_size, S::num_features()]);
        let is_terminal = TensorData::new(is_terminal, [batch_size]);
        let device = B::Device::default();

        TrainingBatch {
            states: Tensor::from_data(states, &device),
            actions: Tensor::from_data(actions, &device),
            invalid_actions_mask: Tensor::from_data(invalid_actions_mask, &device),
            rewards: Tensor::from_data(rewards, &device),
            next_states: Tensor::from_data(next_states, &device),
            is_terminal: Tensor::from_data(is_terminal, &device),
        }
    }
}
