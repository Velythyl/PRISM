
from typing import Union, Type, Optional, Dict, Any

import gym
import numpy as np
import psutil
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv


def _AtariEnv(env_name, n_envs, wrappers_before_atari=[], wrappers_after_atari=[]):
    def make_wrapper(env: gym.Env) -> gym.Env:
        for wrap in wrappers_before_atari:
            env = wrap(env)
        env = AtariWrapper(env, **{})
        for wrap in wrappers_after_atari:
            env = wrap(env)
        return env

    return VecFrameStack(
        make_vec_env(
            env_name,
            n_envs=n_envs,
            seed=None,
            start_index=0,
            wrapper_class=make_wrapper,
        )
        , n_stack=4
    )

def ChromShiftEnv(env_name, n_envs):
    class Shifted(gym.ObservationWrapper):
        def __init__(self, env: gym.Env):
            gym.ObservationWrapper.__init__(self, env)

        def observation(self, frame: np.ndarray) -> np.ndarray:
            frame = frame.dot(np.array([[0.5, .3, .0],
                [.0, 0.6, .1],
                [.1, .0, 0.5]])
            )
            frame = np.clip(frame, 0, 240)
            frame = frame/240 * 255
            frame = frame.astype('uint8')
            import cv2
            cv2.imshow("", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey()
            return frame

    return _AtariEnv(env_name, n_envs, wrappers_before_atari=[Shifted])

def FliplrEnv(env_name, n_envs):
    class Shifted(gym.ObservationWrapper):
        def __init__(self, env: gym.Env):
            gym.ObservationWrapper.__init__(self, env)

        def observation(self, frame: np.ndarray) -> np.ndarray:
            return np.fliplr(frame)

    return _AtariEnv(env_name, n_envs, wrappers_after_atari=[Shifted])


def NormalEnv(env_name, n_envs):
    return _AtariEnv(env_name, n_envs)