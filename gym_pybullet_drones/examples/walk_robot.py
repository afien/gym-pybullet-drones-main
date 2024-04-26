# import gymnasium as gym
# import pybullet
# # import pybullet_envs
# import torch
#
# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
#
#
# # env = gym.make("hover-aviary-v0", render_mode="human")
# env = gym.make("CartPole-v1", render_mode="human")
#
# MAX_AVERAGE_SCORE = 200
#
# policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=[512,512])
# model = PPO("MlpPolicy", env, learning_rate=0.0003, policy_kwargs=policy_kwargs, verbose=1)
#
# for i in range(8000):
#     print("Training iteration ", i)
#     model.learn(total_timesteps=10000)
#     model.save("ppo_Ant_saved_model")
#     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
#     print("mean_reward ", mean_reward)
#     if mean_reward >= MAX_AVERAGE_SCORE:
#         break
#
# del model

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pybullet as p


class QuadrotorEnv(gym.Env):
    def __init__(self):
        super(QuadrotorEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.target_pos = np.array([0, 0, 1])  # Target position (x, y, z)
        self.reset()

    def reset(self):
        self.robot_id = p.loadURDF("plane.urdf", [0, 0, 1])  # Adjust path
        self.current_pos = np.array([0, 0, 1])  # Start position
        return self.current_pos - self.target_pos

    def step(self, action):
        # Apply action to move the quadrotor
        # action: [throttle_1, throttle_2, throttle_3, throttle_4]
        # Implement the logic to move the quadrotor using PyBullet

        # Calculate reward (negative L2 distance to target)
        reward = -np.linalg.norm(self.current_pos - self.target_pos)

        # Check if the quadrotor has reached close to the target position
        done = (np.linalg.norm(self.current_pos - self.target_pos) < 0.1)

        # Return observation, reward, done, info
        return self.current_pos - self.target_pos, reward, done, {}

    def close(self):
        p.disconnect()


env = QuadrotorEnv()
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
