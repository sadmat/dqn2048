use burn::{
    Tensor,
    config::{self, Config},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Float,
};

use crate::dqn::model::Model;

#[derive(Config, Debug)]
pub(crate) struct GameModelConfig {
    // #[config(default = "16")]
    #[config(default = "16 * 11")]
    num_inputs: usize,
    #[config(default = "256")]
    hidden1_size: usize,
    #[config(default = "256")]
    hidden2_size: usize,
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
            output: LinearConfig::new(self.hidden2_size, self.num_outputs).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct GameModel<B: Backend> {
    hidden1: Linear<B>,
    relu1: Relu,
    hidden2: Linear<B>,
    relu2: Relu,
    output: Linear<B>,
}

impl<B: Backend> Model<B> for GameModel<B> {
    fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.hidden1.forward(input);
        let x = self.relu1.forward(x);
        let x = self.hidden2.forward(x);
        let x = self.relu2.forward(x);
        let x = self.output.forward(x);

        x
    }
}
