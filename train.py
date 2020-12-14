import time
from typing import Union, Type, Optional, Dict, Any

import gym
import psutil
import torch
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv

from env import NormalEnv, FliplrEnv
from utils import makedirs

def train(env_name, n_timesteps):

    start = time.time()
    with makedirs(f"./{env_name}/expert"):
        expert = PPO("CnnPolicy",
                     #https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/atari_games.ipynb#scrollTo=TgjfyOTPVxG6
                     FliplrEnv(env_name, psutil.cpu_count(logical=False)-2),#NormalEnv(env_name),
                     #n_steps=64,
                    # batch_size=2048,
                     ##n_epochs=8,
                     #learning_rate=2.5e-4,
                     #clip_range=0.1,
                     #vf_coef=0.5,
                     #ent_coef=0.01,
                     verbose=1
                     ).learn(n_timesteps)
        print(expert.n_envs)
        expert.save(f"./{env_name}/expert/expert_shifted{n_timesteps}")
    print("training took:", time.time()-start)
    env = FliplrEnv(env_name, 1)
    obs = env.reset()
    env.render()
    #time.sleep(100)
    done = False
    trew = 0
    with torch.no_grad():
        while not done:
            #obs = (obs/obs.max()*150).astype(np.uint8) this fucks up the policy
            act = expert.predict(obs)[0]
            #print(act)
            obs, rew, done, _ = env.step(act)
            trew += rew
            env.render()
    print(trew)

if __name__ == "__main__":
    train("PongNoFrameskip-v4", 1000000)