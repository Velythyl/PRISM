import time

import torch
from stable_baselines3 import PPO

from env import EncapsulatePrismEnv
from new_prism import Prism
from prism_bottleneck import PrismBottleneck
from utils import makedirs


def train(env_name, n_timesteps, n_envs, env_class, policy, policy_kwargs=None):
    start = time.time()
    with makedirs(f"./{env_name}/expert"):
        expert = PPO(policy,
                     env_class(env_name, n_envs),
                     policy_kwargs=policy_kwargs,
                     verbose=1
                     )
        expert = expert.learn(n_timesteps)
        expert.save(f"./{env_name}/expert/expert_32{expert.env.classname}_{n_timesteps}")
    print("training took:", time.time()-start)
    return expert

prism = PrismBottleneck(32)
prism.load_state_dict(torch.load(f"{'PongNoFrameskip-v4'}/prism32.pt"))
prism.eval()
prism_env_fun = EncapsulatePrismEnv(prism, is_shifted=False)
charlie = train("PongNoFrameskip-v4", 10000000, 6, prism_env_fun, "MlpPolicy", policy_kwargs={"net_arch": [1000]})
