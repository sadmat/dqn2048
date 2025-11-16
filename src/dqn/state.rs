pub trait ActionType: Clone {
    fn index(&self) -> usize;
}

pub trait StateType: Clone {
    type Action: ActionType;

    fn initial_state() -> Self;
    fn num_actions() -> usize;
    fn num_features() -> usize;
    fn possible_actions(&self) -> Vec<Self::Action>;
    fn advance(&self, action: &Self::Action) -> Self;
    fn is_terminal(&self) -> bool;
    fn as_features(&self) -> Vec<f32>;
}
