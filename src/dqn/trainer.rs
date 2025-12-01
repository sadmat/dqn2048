use std::default::Default;
use std::{cmp::Ordering, f32::NEG_INFINITY};

use burn::{
    Tensor,
    module::AutodiffModule,
    nn::loss::{HuberLossConfig, Reduction::Auto},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, TensorData, backend::AutodiffBackend},
};
use burn::nn::loss::HuberLoss;
use rand::{distr::uniform::SampleRange, rng, rngs::ThreadRng, seq::IndexedRandom, thread_rng};

use crate::dqn::data_augmenter::DataAugmenterType;
use crate::dqn::stats::StatsRecorderType;
use crate::dqn::{
    critic::CriticType,
    model::Model,
    replay_buffer::{ReplayBuffer, StateTransition},
    state::{ActionType, StateType},
};

pub(crate) struct Hyperparameters {
    pub learning_rate: f32,
    pub discount_factor: f32,
    pub exploration_rate: f32,
    pub batch_size: usize,
}

impl Hyperparameters {
    pub(crate) fn new() -> Self {
        Hyperparameters {
            learning_rate: 0.001,
            discount_factor: 0.99,
            exploration_rate: 0.05,
            batch_size: 32,
        }
    }
}

pub(crate) struct Trainer<B, M, S, C, R, D>
where
    B: AutodiffBackend,
    M: Model<B> + AutodiffModule<B>,
    M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
    S: StateType,
    C: CriticType<State = S>,
    R: StatsRecorderType<State = S>,
    D: DataAugmenterType<State = S>,
{
    config: Hyperparameters,
    critic: C,
    replay_buffer: ReplayBuffer<S, D>,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    device: Device<B>,
    stats_recorder: R,
}

impl<B, M, S, C, R, D> Trainer<B, M, S, C, R, D>
where
    B: AutodiffBackend,
    M: Model<B> + AutodiffModule<B>,
    M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
    S: StateType,
    C: CriticType<State = S>,
    R: StatsRecorderType<State = S>,
    D: DataAugmenterType<State = S>,
{
    pub fn new(
        config: Hyperparameters,
        critic: C,
        data_augmenter: D,
        device: Device<B>,
    ) -> Trainer<B, M, S, C, R, D> {
        Trainer {
            config,
            critic: critic,
            replay_buffer: ReplayBuffer::new(data_augmenter),
            optimizer: AdamConfig::new().init(),
            device: device,
            stats_recorder: Default::default(),
        }
    }

    pub fn run_epoch(&mut self, mut model: M) -> (M, R::Stats) {
        let mut state = S::initial_state();

        self.stats_recorder.record_new_epoch();
        while !state.is_terminal() {
            let action = self.pick_action(&state, &model);
            let next_state = state.advance(&action);
            let reward = self.critic.reward(&state, &action, &next_state);
            let transition = StateTransition::new(state, action, reward, next_state.clone());
            self.replay_buffer.store(transition);
            state = next_state;
            self.stats_recorder.record_reward(reward);

            if self.replay_buffer.size() >= self.config.batch_size {
                model = self.training_step(model);
            }
        }
        self.stats_recorder.record_final_state(&state);
        self.stats_recorder
            .record_replay_buffer_size(self.replay_buffer.size());

        let stats = self.stats_recorder.stats();

        (model, stats)
    }

    fn training_step(&mut self, model: M) -> M {
        let huber_loss = HuberLossConfig::new(1.0).init();

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
            - (batch.is_terminal - 1.0) * self.config.discount_factor * target_qvalues;

        let loss = huber_loss.forward(qvalues, target_qvalues, Auto);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        let model = self
            .optimizer
            .step(self.config.learning_rate as f64, model, grads);
        model
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
        let num_features = features.len();
        let data = TensorData::new(features.into(), [1, num_features]);
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
