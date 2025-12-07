use crate::dqn::state::StateType;

pub(crate) trait StatsRecorderType: Default {
    type Stats;
    type State: StateType;

    fn record_new_epoch(&mut self);
    fn record_reward(&mut self, reward: f32);
    fn record_final_state(&mut self, state: &Self::State);
    fn record_replay_buffer_size(&mut self, size: usize);
    fn record_epsilon(&mut self, epsilon: f64);
    fn stats(&self) -> Self::Stats;
}
