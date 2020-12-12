import time

import gym
import psutil
import torch
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack

from env import NormalEnv
from utils import makedirs

#VecFrameStack(make_atari_env(env_name, n_envs=1),n_stack=4)

def train(env_name, n_timesteps):
    start = time.time()
    with makedirs(f"./{env_name}/expert"):
        expert = PPO("CnnPolicy",
                     #https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/atari_games.ipynb#scrollTo=TgjfyOTPVxG6
                     VecFrameStack(make_atari_env(env_name, n_envs=1),n_stack=4),#NormalEnv(env_name),
                     #n_steps=64,
                     #batch_size=512,
                     ##n_epochs=8,
                     #learning_rate=2.5e-4,
                     #clip_range=0.1,
                     #vf_coef=0.5,
                     #ent_coef=0.01,
                     verbose=1
                     ).learn(n_timesteps)
        print(expert.n_envs)
        expert.save(f"./{env_name}/expert/expert_rgb{n_timesteps}")
    print("training took:", time.time()-start)
    env = VecFrameStack(make_atari_env(env_name, n_envs=1),n_stack=4)
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
    train("PongNoFrameskip-v4", 100000)