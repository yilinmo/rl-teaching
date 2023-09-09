from stable_baselines3 import A2C
from gridworldenv import GridworldEnv
import time


env = GridworldEnv(determined=False)
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_gridworld/")
model.learn(total_timesteps=10000)

env = GridworldEnv(determined=False, render_mode= "human")

obs, _info = env.reset()
for i in range(50):
    action, _states = model.predict(obs, deterministic=True)
    obs, _rewards, terminate, _truncate , _info = env.step(action)
    time.sleep(1)
    if terminate == True:
        obs, _info = env.reset()
        time.sleep(1)

