import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import trange

from env import PrismEnv
from new_prism import Prism


def evaluate(env, episodes, policy, render_obs=False):
    if isinstance(env, str):
        env = gym.make(env)

    def render():
        if render_obs:
            env.render()

    all_episode_rewards = []
    for i in trange(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        render()
        while not done:
            action = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            render()
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", episodes)

    return mean_episode_reward

if __name__ == "__main__":
    prism = Prism()
    prism.load_state_dict(torch.load(f"{'PongNoFrameskip-v4'}/new_prism.pt"))
    prism.eval()
    env = PrismEnv(prism, 'PongNoFrameskip-v4', 1, False)

    policy = PPO.load(f"./PongNoFrameskip-v4/expert/expert_NewPrismEnv@NormalEnv_21500000.zip")
    policy.set_env(env)

    evaluate(env, 200, policy, False)