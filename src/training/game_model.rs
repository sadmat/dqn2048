use burn::{
    config::{self, Config},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Float,
    Tensor,
};

use crate::dqn::model::Model;

#[derive(Config, Debug)]
pub(crate) struct GameModelConfig {
    // #[config(default = "16")]
    #[config(default = "16 * 12")]
    num_inputs: usize,
    #[config(default = "768")]
    hidden1_size: usize,
    #[config(default = "512")]
    hidden2_size: usize,
    #[config(default = "256")]
    hidden3_size: usize,
    #[config(default = "4")]
    num_outputs: usize,
}

impl GameModelConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> GameModel<B> {
        GameModel {
            hidden1: LinearConfig::new(self.num_inputs, self.hidden1_size).init(device),
            relu1: Relu::new(),
            hidden2: LinearConfig::new(self.hidden1_size, self.hidden2_size).init(device),
            relu2: Relu::new(),
            hidden3: LinearConfig::new(self.hidden2_size, self.hidden3_size).init(device),
            relu3: Relu::new(),
            value_output: LinearConfig::new(self.hidden3_size, 1).init(device),
            advantage_output: LinearConfig::new(self.hidden3_size, self.num_outputs).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct GameModel<B: Backend> {
    hidden1: Linear<B>,
    relu1: Relu,
    hidden2: Linear<B>,
    relu2: Relu,
    hidden3: Linear<B>,
    relu3: Relu,
    value_output: Linear<B>,
    advantage_output: Linear<B>,
}

impl<B: Backend> Model<B> for GameModel<B> {
    fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.hidden1.forward(input);
        let x = self.relu1.forward(x);
        let x = self.hidden2.forward(x);
        let x = self.relu2.forward(x);
        let x = self.hidden3.forward(x);
        let x = self.relu3.forward(x);

        let state_values = self.value_output.forward(x.clone());
        let advantage_values = self.advantage_output.forward(x);

        let mean_advantage = advantage_values.clone().mean_dim(1);

        state_values + advantage_values - mean_advantage
    }
}
