use crate::dqn::replay_buffer::StateTransition;
use crate::dqn::state::StateType;

pub(crate) trait DataAugmenterType {
    type State: StateType;

    fn augment(&self, transition: StateTransition<Self::State>) -> Vec<StateTransition<Self::State>>;
}