# import random
# import gymnasium as gym

# env = gym.make("CartPole-v1", render_mode="human")
# episodes = 10
# for episode in range(1, episodes+1):
#     observation, info = env.reset()
#     terminated = False
#     score = 0
    
#     while not terminated:
#         action = random.choice([0, 1])
#         observation, reward, terminated, truncated, info = env.step(action)
#         score += reward
#         env.render()

#     print(f"Episode {episode}, score: {score}")

# env.close()


# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

# #train
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading

# #evaluate
# model = PPO.load("ppo_cartpole")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, trunacted, info = env.step(action)
#     env.render()


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

env_name = "LunarLander-v2"
env = gym.make(env_name)
env = DummyVecEnv([lambda : env])

model = PPO("MlpPolicy",
            env=env,
            batch_size=64,
            gae_lambda=0.98,
            gamma=0.999,
            n_epochs=4,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tensorboard/LunarLander-v2/")

model.learn(total_timesteps=10000)
model.save("./model/LunarLander_PPO.pk1")

env = gym.make(env_name)
model = PPO.load("./model/LunarLander_PPO.pk1")
state = env.reset()
done = False
score = 0
while not done:
    action,_ = model.predict(observation=state)
    state, reward, done, info = env.step(action=action)
    score += reward
    env.render()
env.close()
print(score)