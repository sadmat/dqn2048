use crate::dqn::state::StateType;

pub(crate) trait CriticType: Default {
    type State: StateType;

    fn reward(
        &self,
        state: &Self::State,
        action: &<Self::State as StateType>::Action,
        next_state: &Self::State,
    ) -> f32;
}
