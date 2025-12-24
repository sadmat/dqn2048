use crate::dqn::replay_buffer::StateTransition;
use crate::dqn::state::StateType;

pub(crate) trait DataAugmenterType {
    type State: StateType;

    fn augment(&self, state: Self::State, action: <Self::State as StateType>::Action, reward: f32, next_state: Self::State) -> Vec<StateTransition>;
}