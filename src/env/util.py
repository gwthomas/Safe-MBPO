from gym.spaces import Box, Discrete
import numpy as np


def isbox(space):
    return isinstance(space, Box)

def isdiscrete(space):
    return isinstance(space, Discrete)


def space_dim(space):
    if isbox(space):
        return int(np.prod(space.shape))
    elif isdiscrete(space):
        return space.n
    else:
        raise ValueError(f'Unknown space {space}')


def env_dims(env):
    return (space_dim(env.observation_space), space_dim(env.action_space))


def get_max_episode_steps(env):
    if hasattr(env, '_max_episode_steps'):
        return env._max_episode_steps
    elif hasattr(env, 'env'):
        return get_max_episode_steps(env.env)
    else:
        raise ValueError('env does not have _max_episode_steps')


def get_done(env):
    if hasattr(env.__class__, 'done'):
        return env.__class__.done
    else:
        return get_done(env.env)