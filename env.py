import inspect
from typing import Union, Type, Optional, Dict, Any

import gym
import numpy as np
import psutil
import torch
from gym.wrappers import FrameStack, AtariPreprocessing
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv
from torchvision.transforms import ToTensor

from new_prism import Prism


def _AtariEnv(env_name, n_envs, seed=None, wrappers_before_atari=[], wrappers_after_atari=[]):
    def make_wrapper(env: gym.Env) -> gym.Env:
        for wrap in wrappers_before_atari:
            env = wrap(env)
        env = AtariWrapper(env, **{})
        for wrap in wrappers_after_atari:
            env = wrap(env)
        return env

    temp = make_vec_env(
            env_name,
            n_envs=n_envs,
            seed=seed,
            start_index=0,
            wrapper_class=make_wrapper,
        )
    env = VecFrameStack(
        make_vec_env(
            env_name,
            n_envs=n_envs,
            seed=seed,
            start_index=0,
            wrapper_class=make_wrapper,
        )
        , n_stack=4
    )
    env.classname = inspect.currentframe().f_back.f_code.co_name
    return env

def ChromShiftEnv(env_name, n_envs, seed=None):
    filter = np.array([[.393, .769, .189],
                         [.349, .686, .168],
                         [.272, .534, .131]]).T
    class Shifted(gym.ObservationWrapper):
        def __init__(self, env: gym.Env):
            gym.ObservationWrapper.__init__(self, env)
        def observation(self, frame: np.ndarray) -> np.ndarray:
            frame = frame.astype("float64")
            np.dot(frame, filter, out=frame)
            np.clip(frame, 0, 240, out=frame)
            frame *= 240/255
            frame = frame.astype('uint8')
            return frame

    return _AtariEnv(env_name, n_envs, wrappers_before_atari=[Shifted], seed=seed)

def PrismEnv(prism, env_name, n_envs, is_shifted=False, seed=None):
    env = NormalEnv(env_name, n_envs, seed=seed) if not is_shifted else ChromShiftEnv(env_name, n_envs, seed=seed)

    prism.to("cuda")

    step_wait_copy = env.step_wait
    reset_copy = env.reset

    to_tensor = ToTensor()
    def apply_prism(vectorized_stacked_frames):
        shape = vectorized_stacked_frames.shape
        tensor = torch.zeros((shape[0], shape[-1], shape[1], shape[2])).float()

        for i, frame_stack in enumerate(vectorized_stacked_frames):
            tensorized = to_tensor(frame_stack).float()
            tensor[i] = tensorized
        with torch.no_grad():
            prismed = prism(tensor.to("cuda"))
            prismed /= prismed.max()
            #prismed *= 100
            #torch.round(prismed, out=prismed)
            #prismed /= 100
        prismed = prismed.cpu().numpy()

        return prismed

    def step_wait():
        stackedobs, rewards, dones, infos = step_wait_copy()
        return apply_prism(stackedobs), rewards, dones, infos

    def reset():
        stackedobs = reset_copy()
        return apply_prism(stackedobs)

    env.width = None
    env.height = None
    env.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(512 if isinstance(prism, Prism) else prism.neck,), dtype=np.float
    )

    env.reset = reset
    env.step_wait = step_wait

    env.classname = "PrismEnv@"+env.classname

    return env

def EncapsulatePrismEnv(prism, is_shifted=False):
    def fun(env_name, n_envs):
        return PrismEnv(prism, env_name, n_envs, is_shifted)
    return fun

def NormalEnv(env_name, n_envs, seed=None):
    return _AtariEnv(env_name, n_envs, seed=seed)