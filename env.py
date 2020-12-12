import gym
import numpy as np
import psutil
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from atari_wrapper import AtariWrapper


def NormalEnv(env, n_envs=psutil.cpu_count(logical=False) - 2):
    if isinstance(env, str):
        env = gym.make(env)

    def make_atari_env(env_id,seed=None):
        wrapper_kwargs = {}

        def atari_wrapper(env: gym.Env) -> gym.Env:
            env = AtariWrapper(env, **wrapper_kwargs)   # our own wrapper, not stable_baselines'
            return env

        return make_vec_env(
            env_id,
            n_envs=n_envs,
            seed=seed,
            wrapper_class=atari_wrapper
        )

    def fake_class():
        return env

    return VecFrameStack(
        make_atari_env(
            fake_class
        ),
        n_stack=4
    )

def ShiftedEnv(env, filter):
    if isinstance(env, str):
        env = gym.make(env)

    reset_fun = env.reset
    step_fun = env.step

    def apply_filter(obs):
        obs = obs.dot(filter)
        obs /= obs.max()
        return obs.astype(np.float32)

    def reset():
        obs = reset_fun()
        return apply_filter(obs)

    def step(act):
        obs, rew, done, info = step_fun(act)
        return apply_filter(obs), rew, done, info

    env.reset = reset
    env.step = step

    return env

def PrismEnv(env, prism):
    if isinstance(env, str):
        env = gym.make(env)

    reset_fun = env.reset
    step_fun = env.step

    def reset():
        obs = reset_fun()
        return prism(obs)

    def step(act):
        obs, rew, done, info = step_fun(act)
        return prism(obs), rew, done, info

    env.reset = reset
    env.step = step

    return env