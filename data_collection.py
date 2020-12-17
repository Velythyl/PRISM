import cv2
import torch
from stable_baselines3 import PPO
from torchvision.transforms import ToTensor

from env import NormalEnv, FliplrEnv, ChromShiftEnv
from tqdm import trange
import numpy as np

from prism import Prism
from utils import makedirs


def not_main(env_name):
    env = NormalEnv(env_name, 1, seed=0)
    shifted = ChromShiftEnv(env_name, 1)

    policy = PPO.load(f"./{env_name}/expert/expert_NormalEnv_1000000.zip")
    policy.set_env(env)
    shifted_policy = PPO.load(f"./{env_name}/expert/expert_ChromShiftEnv_1000000.zip")
    shifted_policy.set_env(shifted)

    obs = env.reset()[0]
    s_obs = shifted.reset()[0]

    prism = Prism()
    prism.load_state_dict(torch.load(f"./{env_name}/prism.pt"))
    prism.eval()

    with torch.no_grad():
        obs = torch.unsqueeze(ToTensor()(obs).float(), 0)
        s_obs = torch.unsqueeze(ToTensor()(s_obs).float(), 0)

        x = prism(obs)
        y = prism(s_obs)
        print(torch.max(torch.abs_(x-y)))
        print(torch.max(x))



    aaa=0



def main(env_name, nb_to_collect):
    env = NormalEnv(env_name, 1)
    shifted = ChromShiftEnv(env_name, 1)

    policy = PPO.load(f"./{env_name}/expert/expert_NormalEnv_1000000.zip")
    policy.set_env(env)
    shifted_policy = PPO.load(f"./{env_name}/expert/expert_ChromShiftEnv_1000000.zip")
    shifted_policy.set_env(shifted)

    old_env_obs = [env.reset(),False]
    old_shifted_obs = [shifted.reset(),False]

    def mkfun(sim, expert, buffer):
        def step():
            if buffer[1]:
                buffer[0] = sim.reset()

            act = expert.predict(buffer[0], deterministic=True)[0]
            obs, _, done, _ = sim.step(act)

            #cv2.imshow("a", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            #cv2.waitKey()

            #print(sim.classname, obs.shape)

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

if __name__ == "__main__":
    #main("PongNoFrameskip-v4", 20000)
    #main("PongNoFrameskip-v4", 20000)
    not_main("PongNoFrameskip-v4")