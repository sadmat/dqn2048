use crate::dqn::replay_buffer::StateTransition;
use burn::prelude::TensorData;
use burn::tensor::backend::AutodiffBackend;
use burn::{
    tensor::{Bool, Float, Int},
    Tensor,
};

pub struct TrainingBatch<B: AutodiffBackend> {
    pub states: Tensor<B, 2, Float>,
    pub actions: Tensor<B, 1, Int>,
    pub invalid_actions_mask: Tensor<B::InnerBackend, 2, Bool>,
    pub rewards: Tensor<B::InnerBackend, 1, Float>,
    pub next_states: Tensor<B::InnerBackend, 2, Float>,
    pub is_terminal: Tensor<B::InnerBackend, 1, Float>,
    pub weights: Tensor<B, 1, Float>,
    pub indices: Vec<usize>,
}

impl<B: AutodiffBackend> TrainingBatch<B> {
    pub(crate) fn from(
        records: Vec<&StateTransition>,
        weights: Vec<f32>,
        indices: Vec<usize>,
    ) -> Self {
        let batch_size = records.len();
        let state_size = records[0].state.len();
        let actions_size = records[0].invalid_actions_mask.len();

        let mut states: Vec<f32> = Vec::with_capacity(batch_size * state_size);
        let mut actions: Vec<i32> = Vec::with_capacity(batch_size);
        let mut invalid_actions_mask: Vec<bool> = Vec::with_capacity(batch_size * actions_size);
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut next_states: Vec<f32> = Vec::with_capacity(batch_size * state_size);
        let mut is_terminal: Vec<f32> = Vec::with_capacity(batch_size);

        for record in records {
            states.extend_from_slice(record.state.as_slice());
            actions.push(record.action as i32);
            invalid_actions_mask.extend_from_slice(record.invalid_actions_mask.as_slice());
            rewards.push(record.reward);
            next_states.extend_from_slice(record.next_state.as_slice());
            is_terminal.push(record.is_terminal);
        }

        let states = TensorData::new(states, [batch_size, state_size]);
        let actions = TensorData::new(actions, [batch_size]);
        let invalid_actions_mask =
            TensorData::new(invalid_actions_mask, [batch_size, actions_size]);
        let rewards = TensorData::new(rewards, [batch_size]);
        let next_states = TensorData::new(next_states, [batch_size, state_size]);
        let is_terminal = TensorData::new(is_terminal, [batch_size]);
        let weights = TensorData::new(weights, [batch_size]);
        let device = B::Device::default();

        TrainingBatch {
            states: Tensor::from_data(states, &device).detach(),
            actions: Tensor::from_data(actions, &device),
            invalid_actions_mask: Tensor::from_data(invalid_actions_mask, &device),
            rewards: Tensor::from_data(rewards, &device),
            next_states: Tensor::from_data(next_states, &device),
            is_terminal: Tensor::from_data(is_terminal, &device),
            weights: Tensor::from_data(weights, &device).detach(),
            indices,
        }
    }
}
