use crate::dqn::data_augmenter::DataAugmenterType;
use crate::dqn::stats::StatsRecorderType;
use crate::dqn::{
    critic::CriticType,
    model::Model,
    replay_buffer::ReplayBuffer,
    state::{ActionType, StateType},
};
use burn::{
    module::AutodiffModule,
    nn::loss::{HuberLossConfig, Reduction::Auto},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Device, TensorData},
    Tensor,
};
use rand::{distr::uniform::SampleRange, rng, rngs::ThreadRng, seq::IndexedRandom};
use std::default::Default;
use std::{cmp::Ordering, f32::NEG_INFINITY};

pub(crate) struct Hyperparameters {
    pub learning_rate: f32,
    pub discount_factor: f32,
    pub batch_size: usize,
    pub replay_buffer_capacity: usize,
    pub per_alpha: f32,
    pub per_beta: f32,
    pub per_beta_increment: f32,
    pub per_epsilon: f32,
    pub initial_epsilon: f64,
    pub epsilon_decay_frames: i32,
    pub min_epsilon: f64,
    pub training_frequency: usize,
    pub network_sync_frequency: usize,
}

impl Hyperparameters {
    pub(crate) fn new() -> Self {
        Hyperparameters {
            learning_rate: 0.00025,
            discount_factor: 0.99,
            batch_size: 8 * 1024,
            replay_buffer_capacity: 2_u32.pow(20) as usize,
            per_alpha: 0.6,
            per_beta: 0.4,
            per_beta_increment: 0.001,
            per_epsilon: 0.001,
            initial_epsilon: 0.5,
            epsilon_decay_frames: 7500,
            min_epsilon: 0.0001,
            training_frequency: 25,
            network_sync_frequency: 10000,
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
    epoch_num: usize,
    frame_num: usize,
    target_network: Option<M>,
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
        let replay_buffer_capacity = config.replay_buffer_capacity;

        Trainer {
            config,
            critic: critic,
            replay_buffer: ReplayBuffer::new(data_augmenter, replay_buffer_capacity),
            optimizer: AdamConfig::new().init(),
            device: device,
            stats_recorder: Default::default(),
            epoch_num: 0,
            frame_num: 0,
            target_network: None,
        }
    }

    pub fn run_epoch(&mut self, mut model: M) -> (M, R::Stats) {
        // Epoch initialization

        let mut state = S::initial_state();
        let mut epoch_frames = 0;
        self.epoch_num += 1;
        let epsilon = f64::max(
            self.config.min_epsilon,
            self.config.initial_epsilon
                * (self.config.epsilon_decay_frames - self.epoch_num as i32) as f64
                / self.config.epsilon_decay_frames as f64,
        );
        if self.target_network.is_none() {
            self.target_network = Some(model.clone());
        }

        // Epoch loop

        self.stats_recorder.record_new_epoch();
        self.stats_recorder.record_epsilon(epsilon);
        while !state.is_terminal() {
            self.frame_num += 1;
            let action = self.pick_action(&state, &model, epsilon);
            let next_state = state.advance(&action);
            let reward = self.critic.reward(&state, &action, &next_state);
            self.replay_buffer
                .store(state, action, reward, next_state.clone());
            state = next_state;
            self.stats_recorder.record_reward(reward);
            epoch_frames += 1;

            if self.replay_buffer.size() >= self.config.batch_size
                && self.frame_num % self.config.training_frequency == 0
            {
                model = self.training_step(model);
            }
            if self.frame_num % self.config.network_sync_frequency == 0 {
                self.target_network = Some(model.clone());
            }
        }
        self.stats_recorder.record_final_state(&state, epoch_frames);
        self.stats_recorder
            .record_replay_buffer_size(self.replay_buffer.size());

        let stats = self.stats_recorder.stats();

        (model, stats)
    }

    fn training_step(&mut self, model: M) -> M {
        let huber_loss = HuberLossConfig::new(1.0).init();
        let Some(target_network) = self.target_network.as_ref() else {
            panic!("Target network should've been set by run_epoch()")
        };
        let per_beta = (self.config.per_beta
            + self.config.per_beta_increment * self.epoch_num as f32)
            .min(1.0);

        let batch = self.replay_buffer.sample(self.config.batch_size, per_beta);
        let output = model.forward(batch.states);
        let qvalues: Tensor<B, 1> = output
            .gather(1, batch.actions.unsqueeze_dim(1))
            .squeeze_dim(1);
        let target_qvalues = target_network
            .valid()
            .forward(batch.next_states)
            .mask_fill(batch.invalid_actions_mask, NEG_INFINITY)
            .max_dim(1)
            .squeeze_dim(1);
        let target_qvalues = batch.rewards
            + (1.0 - batch.is_terminal) * self.config.discount_factor * target_qvalues;
        let target_qvalues = Tensor::from_inner(target_qvalues).detach();

        let td_errors = (qvalues.clone() - target_qvalues.clone()).abs();
        let td_errors = td_errors.into_data().into_vec().unwrap();
        self.replay_buffer.update_priorities(
            batch.indices.as_slice(),
            td_errors.as_slice(),
            self.config.per_alpha,
            self.config.per_epsilon,
        );

        let loss = huber_loss.forward(qvalues, target_qvalues, Auto);
        let loss = loss * batch.weights;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        let model = self
            .optimizer
            .step(self.config.learning_rate as f64, model, grads);
        model
    }

    fn pick_action(&self, state: &S, model: &M, epsilon: f64) -> S::Action {
        let mut rng = rng();
        if (0.0..=1.0).sample_single(&mut rng).unwrap() <= epsilon {
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
