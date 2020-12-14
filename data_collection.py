from stable_baselines3 import PPO

from env import NormalEnv, FliplrEnv
from tqdm import trange
import numpy as np

from utils import makedirs


def main(env_name, nb_to_collect):
    env = NormalEnv(env_name, 1)
    shifted = FliplrEnv(env_name, 1)

    policy = PPO.load(f"./{env_name}/expert/expert_nonshifted1000000.zip")
    policy.set_env(env)
    shifted_policy = PPO.load(f"./{env_name}/expert/expert_shifted1000000.zip")
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

    obs_data = np.zeros((nb_to_collect, 84, 84, 4))
    act_data = np.zeros((nb_to_collect,))

    for i in trange(0, nb_to_collect, 2):
        for j, f in enumerate(funs):
            obs, act = f()
            obs_data[i+j] = obs
            act_data[i+j] = act

    with makedirs(f"./{env_name}/dataset"):
        np.savez_compressed(f"./{env_name}/dataset/dataset.npz", a=obs_data, b=act_data)

if __name__ == "__main__":
    main("PongNoFrameskip-v4", 20000)