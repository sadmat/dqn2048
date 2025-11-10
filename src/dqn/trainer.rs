use std::{cmp::Ordering, f32::NEG_INFINITY};

use burn::{
    Tensor,
    module::AutodiffModule,
    nn::loss::{HuberLossConfig, Reduction::Auto},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, TensorData, backend::AutodiffBackend},
};
use rand::{distr::uniform::SampleRange, rng, rngs::ThreadRng, seq::IndexedRandom, thread_rng};

use crate::dqn::{
    model::Model,
    replay_buffer::{ReplayBuffer, StateTransition},
    state::{ActionType, State},
};

pub(crate) struct Hyperparameters {
    pub learning_rate: f32,
    pub discount_factor: f32,
    pub exploration_rate: f32,
    pub batch_size: usize,
}

pub(crate) struct Trainer<B, M, const N: usize, S, R>
where
    B: AutodiffBackend,
    M: Model<B> + AutodiffModule<B>,
    M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
    S: State<N>,
    R: Fn(&S, &S::Action, &S) -> f32,
{
    config: Hyperparameters,
    reward: R,
    replay_buffer: ReplayBuffer<N, S>,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    device: Device<B>,
}

impl<B, M, const N: usize, S, R> Trainer<B, M, N, S, R>
where
    B: AutodiffBackend,
    M: Model<B> + AutodiffModule<B>,
    M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
    S: State<N>,
    R: Fn(&S, &S::Action, &S) -> f32,
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
            let action = self.pick_action(&state, &model);
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

    fn pick_action(&self, state: &S, model: &M) -> S::Action {
        let mut rng = rng();
        if (0.0..=1.0).sample_single(&mut rng).unwrap() <= self.config.exploration_rate {
            self.pick_random_action(state, &mut rng)
        } else {
            self.pick_best_action(state, model)
        }
    }

    fn pick_random_action(&self, state: &S, rng: &mut ThreadRng) -> S::Action {
        state.possible_actions().choose(rng).unwrap().clone()
    }

    fn pick_best_action(&self, state: &S, model: &M) -> S::Action {
        let features = state.as_features();
        let data = TensorData::new(features.into(), S::features_shape());
        let input = Tensor::<B::InnerBackend, 2>::from_data(data, &self.device);
        let output = model.valid().forward(input);
        let output: Vec<f32> = output.into_data().into_vec().unwrap();

        let best_action = state
            .possible_actions()
            .into_iter()
            .map(|action| {
                let index = action.index();
                (action, output[index])
            })
            .max_by(|lhs, rhs| {
                if lhs.1 > rhs.1 {
                    Ordering::Greater
                } else if lhs.1 < rhs.1 {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .unwrap();

        best_action.0
    }
}
