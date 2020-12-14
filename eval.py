import gym
import numpy as np

def evaluate(env, episodes, policy, render_obs=False):
    if isinstance(env, str):
        env = gym.make(env)

    def render():
        if render_obs:
            env.render()

    all_episode_rewards = []
    for i in range(episodes):
        print(i)
        episode_rewards = []
        done = False
        obs = env.reset()
        render()
        while not done:
            # _states are only useful when using LSTM policies
            action = policy.predict(obs, deterministic=True)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            render()
            episode_rewards.append(reward)
        print(sum(episode_rewards))

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", episodes)

    return mean_episode_reward
