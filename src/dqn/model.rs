use burn::{
    Tensor,
    module::AutodiffModule,
    tensor::{Float, backend::AutodiffBackend},
};

pub trait Model<B: AutodiffBackend>: AutodiffModule<B> {
    fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float>;
}
