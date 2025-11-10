use std::f32::NEG_INFINITY;

use burn::{
    Tensor,
    nn::loss::{HuberLossConfig, Reduction::Auto},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, backend::AutodiffBackend},
};

use crate::dqn::{
    model::Model,
    replay_buffer::{ReplayBuffer, StateTransition},
    state::State,
};

pub(crate) struct Hyperparameters {
    pub learning_rate: f32,
    pub discount_factor: f32,
    pub exploration_rate: f32,
    pub batch_size: usize,
}

pub(crate) struct Trainer<
    B: AutodiffBackend,
    M: Model<B>,
    const N: usize,
    S: State<N>,
    R: Fn(&S, &S::Action, &S) -> f32,
> {
    config: Hyperparameters,
    reward: R,
    replay_buffer: ReplayBuffer<N, S>,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    device: Device<B>,
}

impl<B: AutodiffBackend, M: Model<B>, const N: usize, S: State<N>, R: Fn(&S, &S::Action, &S) -> f32>
    Trainer<B, M, N, S, R>
{
    pub fn new(config: Hyperparameters, reward: R, device: Device<B>) -> Trainer<B, M, N, S, R> {
        Trainer {
            config,
            reward: reward,
            replay_buffer: ReplayBuffer::new(),
            optimizer: AdamConfig::new().init(),
            device: device,
        }
    }

    pub fn run_epoch(&mut self, model: M) -> M {
        let huber_loss = HuberLossConfig::new(1.0).init();
        let mut state = S::initial_state();

        while !state.is_terminal() {
            let action = self.pick_action(&state);
            let next_state = state.advance(&action);
            let reward = (self.reward)(&state, &action, &next_state);
            let transition = StateTransition::new(state, action, reward, next_state.clone());
            self.replay_buffer.store(transition);
            state = next_state;
        }

        if self.replay_buffer.size() < self.config.batch_size {
            return model;
        }

        let batch = self.replay_buffer.sample(self.config.batch_size);
        let output = model.forward(batch.states);
        let qvalues: Tensor<B, 1> = output
            .gather(1, batch.actions.unsqueeze_dim(1))
            .squeeze_dim(1);
        let target_qvalues = model
            .forward(batch.next_states)
            .detach()
            .mask_fill(batch.invalid_actions_mask, NEG_INFINITY)
            .max_dim(1)
            .squeeze_dim(1);
        let target_qvalues = batch.rewards
            + (batch.is_terminal - 1.0) * self.config.discount_factor * target_qvalues;

        let loss = huber_loss.forward(qvalues, target_qvalues, Auto);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        self.optimizer
            .step(self.config.learning_rate as f64, model, grads)
    }

    fn pick_action(&self, state: &S) -> S::Action {
        todo!();
    }
}
