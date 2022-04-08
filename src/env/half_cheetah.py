from gym.envs.mujoco import HalfCheetahEnv as GymHalfCheetahEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper


class HalfCheetahEnv(GymHalfCheetahEnv, MujocoWrapper):
    def __init__(self):
        super().__init__()
        self._max_episode_steps = 1000

    @staticmethod
    def check_done(states):
        return np.zeros(len(states), dtype=bool)

    def qposvel_from_obs(self, obs):
        qpos = self.sim.data.qpos.copy()
        qpos[1:] = obs[:8]
        qvel = obs[8:]
        return qpos, qvel