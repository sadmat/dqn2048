pub trait ActionType: Clone {
    fn index(&self) -> usize;
}

pub trait StateType: Clone {
    type Action: ActionType;

    const NUM_ACTIONS: usize;
    const NUM_FEATURES: usize;

    fn initial_state() -> Self;
    fn possible_actions(&self) -> Vec<Self::Action>;
    fn advance(&self, action: &Self::Action) -> Self;
    fn is_terminal(&self) -> bool;
    fn as_features(&self) -> Vec<f32>;
}
