import random

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn
from torch.nn.functional import mse_loss

from env import PrismEnv, NormalEnv
from new_prism import Prism
from prism_bottleneck import PrismBottleneck

prism = PrismBottleneck(32)
prism.load_state_dict(torch.load(f"{'PongNoFrameskip-v4'}/prism32.pt"))
prism.eval()

def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

seed(0)
original_env = NormalEnv("PongNoFrameskip-v4", 1, seed=0)
obs = original_env.reset()
seed(0)
a = PrismEnv(prism, "PongNoFrameskip-v4", 1, is_shifted=False, seed=0)
a_obs = a.reset()
seed(0)
b = PrismEnv(prism, "PongNoFrameskip-v4", 1, is_shifted=True, seed=0)
b_obs = b.reset()

all_m = []

def print_max_abs_diff(x,y):
    # [0] to get the array, x and y are 1-tuples
    x = x[0]
    y = y[0]

    mse = np.sum(np.abs(x-y))

    temp_m = np.abs(x-y)
    m = np.max(temp_m)
    #print("M:", m, "l:", mse, "number non-zero:", np.count_nonzero(temp_m))
    all_m.append(m)
    return m

print_max_abs_diff(a_obs, b_obs)

# we can't just make Alice and Bob play separately as we have no guarantees that both will act the exact same way.
# instead, we'll arbitrarily have just Alice play on both using the actions she would output on her original env.

alice = PPO.load(f"./PongNoFrameskip-v4/expert/expert_NormalEnv_1000000.zip")
alice.set_env(original_env)

done = False
while not done:
    seed(0)
    act = alice.predict(obs, deterministic=True)
    seed(0)
    obs, _, done, _ = original_env.step(act)

    seed(0)
    obsa, _, done1, _ = a.step(act)
    seed(0)
    obsb, _, done2, _ = b.step(act)

    print(done1)
    print(done2)

    print_max_abs_diff(obsa, obsb)

print(np.mean(np.array(all_m)))