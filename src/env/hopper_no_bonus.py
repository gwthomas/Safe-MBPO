import numpy as np
from force.env.mujoco.hopper import HopperEnv


class HopperNoBonusEnv(HopperEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        reward -= 1     # subtract out alive bonus
        info['violation'] = done
        return next_state, reward, done, info

    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        heights, angs = states[:,0], states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))