import gym
import time

import psutil
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from env import NormalEnv, FliplrEnv, PrismEnv, ChromShiftEnv
from eval import evaluate
from prism import Prism, Head, PrismAndHead
from torchvision.transforms import ToTensor
import cv2



def get_fake_policy(env):
    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            prism = Prism()
            head = Head(env.action_space.n)
            self.pah = PrismAndHead(prism, head)
            self.pah.eval()

        def predict(self, obs):
            with torch.no_grad():
                tensor_obs = ToTensor()(obs)
                tensor_obs = torch.unsqueeze(tensor_obs, 0)
                return np.argmax(self.pah(tensor_obs.to("cuda")).cpu().numpy())
    policy = FakePolicy()
    policy = policy.to("cuda")
    return policy

def test_loop(env_name, policy=None):
    #prism = Prism()
    #prism.load_state_dict(torch.load(f"{env_name}/prism.pt"))
    #prism.eval()
    env = ChromShiftEnv(env_name, 1)
    #env = ShiftedEnv(env_name, np.array([[.13, 0, 0],
                           #  [0, .1, 0],
                            # [0 ,0 ,.1]]))

    policy = PPO.load(f"./{env_name}/expert/expert_ChromShiftEnv_1000000.zip")
    policy.set_env(env)

    if policy is None:
        policy = get_fake_policy(env)

    evaluate(env, 200, policy, True)
    exit()

    obs = env.reset()
    #obs1 = env1.reset()

#$cv2.imshow("a", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)); cv2.waitKey()
 #   cv2.imshow("a", cv2.cvtColor(obs1, cv2.COLOR_RGB2BGR)); cv2.waitKey()

    env.render()
    done = False
    trew = 0
    with torch.no_grad():
        while not done:
            act = policy.predict(obs)
            print(act)
            obs, rew, done, _ = env.step(act)
            env.render()
            #time.sleep(1)
            trew += rew
    print("Done testing.", trew)

if __name__ == "__main__":
    test_loop('PongNoFrameskip-v4')