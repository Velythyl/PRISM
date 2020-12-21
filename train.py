import time

import gym
import psutil
import torch
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv

from env import NormalEnv, FliplrEnv, PrismEnv, ChromShiftEnv, TempEnv
from prism import Prism
from utils import makedirs

def train(env_name, n_timesteps):
    prism = Prism()
    prism.load_state_dict(torch.load(f"./{env_name}/prism16.pt"))
    prism.eval()

    x = NormalEnv(env_name, 6)
    y = PrismEnv(NormalEnv(env_name, 6), prism)
    z = PrismEnv(ChromShiftEnv(env_name, 6), prism)

    a = x.reset()
    b = y.reset()
    c = z.reset()

    temp = b.astype(float)-c
    temp2 = temp.max()

    surrogate = gym.make("Pong-ram-v4")


    #env = PrismEnv(NormalEnv(env_name, n_envs=psutil.cpu_count(logical=False)-2),
     #                         prism
     #                         )
   # obs = env.reset()


#    z = ChromShiftEnv(env_name, n_envs=1).classname
    start = time.time()
    with makedirs(f"./{env_name}/expert"):
        expert = PPO("MlpPolicy",
                     #https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/atari_games.ipynb#scrollTo=TgjfyOTPVxG6
                     PrismEnv(NormalEnv(env_name, 6), prism),

                     #policy_kwargs={"net_arch": [dict(pi=[64,64,64], vf=[64,64,64])], "activation_fn": torch.nn.LeakyReLU},
                     n_steps=10000,
                     # batch_size=2048,
                     ##n_epochs=8,
                     #learning_rate=2.5e-4,
                     #clip_range=0.1,
                     #vf_coef=0.5,
                     #ent_coef=0.01,
                     verbose=1
                     ).learn(n_timesteps)
        expert.save(f"./{env_name}/expert/expert_latest2_{n_timesteps}")
    print("training took:", time.time()-start)
    env = PrismEnv(NormalEnv(env_name, 1), prism)
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
    train("PongNoFrameskip-v4", 15000000)
