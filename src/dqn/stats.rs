use crate::dqn::state::StateType;

pub(crate) trait StatsRecorderType: Default {
    type Stats;
    type State: StateType;

    fn record_new_epoch(&mut self);
    fn record_reward(&mut self, reward: f32);
    fn record_final_state(&mut self, state: &Self::State);
    fn stats(&self) -> Self::Stats;
}