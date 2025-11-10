use burn::{
    Tensor,
    module::{AutodiffModule, Module},
    prelude::Backend,
    tensor::{Float, backend::AutodiffBackend},
};

pub trait Model<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float>;
}
