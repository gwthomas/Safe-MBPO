from abc import ABC, abstractmethod

import numpy as np
from gym.envs.mujoco import MujocoEnv


class MujocoWrapper(ABC, MujocoEnv):
    @abstractmethod
    def qposvel_from_obs(self, obs):
        pass

    def set_state_from_obs(self, obs):
        qpos, qvel = self.qposvel_from_obs(obs)
        self.set_state(qpos, qvel)

    def oracle_step(self, state, action):
        try:
            self.set_state_from_obs(state)
            next_state, reward, done, info = self.step(action)
        except:
            next_state = np.full_like(state, np.nan)
            reward = np.nan
            done = True
            info = {'Unstable dynamics': True}
        return next_state, reward, done, info

    def oracle_dynamics(self, state, action):
        next_state, reward, _, _ = self.oracle_step(state, action)
        return next_state, reward