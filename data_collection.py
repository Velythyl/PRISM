import cv2
import torch
from stable_baselines3 import PPO
from torchvision.transforms import ToTensor

from env import NormalEnv, FliplrEnv, ChromShiftEnv
from tqdm import trange
import numpy as np

from new_prism import Prism
from utils import makedirs

def data_collection(env_name, alice, bob, alice_env, bob_env, nb_to_collect):
    env = alice_env
    shifted = bob_env

    policy = alice
    policy.set_env(env)
    shifted_policy = bob
    shifted_policy.set_env(shifted)

    old_env_obs = [env.reset(),False]
    old_shifted_obs = [shifted.reset(),False]

    def mkfun(sim, expert, buffer):
        def step():
            if buffer[1]:
                buffer[0] = sim.reset()

            act = expert.predict(buffer[0], deterministic=True)[0]
            obs, _, done, _ = sim.step(act)

            ret_obs = buffer[0]

            buffer[0] = obs
            buffer[1] = done

            return ret_obs, act
        return step

    funs = (
        mkfun(env, policy, old_env_obs),
        mkfun(shifted, shifted_policy, old_shifted_obs)
    )

    obs_data = []
    act_data = []

    for i in trange(0, nb_to_collect, 2):
        for j, f in enumerate(funs):
            obs, act = f()
            obs_data.append(obs[0])
            act_data.append(act[0])
    obs_data = np.array(obs_data)
    act_data = np.array(act_data)


    with makedirs(f"./{env_name}/dataset"):
        np.savez_compressed(f"./{env_name}/dataset/dataset.npz", a=obs_data, b=act_data)