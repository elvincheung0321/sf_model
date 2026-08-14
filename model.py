import stable_retro as retro
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO



# env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis-v0")



class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
        self.action_space = MultiBinary(12)

        self.game = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis-v0", use_restricted_actions=retro.Actions.FILTERED, render_mode=None)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        obs = self.preprocess(obs)

        self.previous_frame = obs

        if info["health"] == 0 and info["enemy_health"] == 0:
            reward = 0
            self.enemy_health = 0
            self.player_health = 0
        else:
            dmg_dealt = self.enemy_health -info["enemy_health"]
            dmg_taken = self.player_health- info["health"]
            self.enemy_health = info["enemy_health"]
            self.player_health = info["health"]
            reward = dmg_dealt - dmg_taken
        

        return obs, reward, terminated, truncated, info


    def render(self, *args, **kwargs):
        self.game.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.game.reset(seed=seed, options=options)
        obs = self.preprocess(obs)
        self.previous_frame = obs

        
        info = self.game.data.lookup_all()
        self.player_health = info.get("health", 0)
        self.enemy_health = info.get("enemy_health", 0)
        return obs, info

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)

        channels = np.reshape(resize, (84,84,1))
        return channels


    def close(self):
        self.game.close()

LOG_DIR = "./opt_logs/"


def make_env():
    return Monitor(StreetFighter(), LOG_DIR)


def main():
    env = SubprocVecEnv([make_env for _ in range(4)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    model_params = {
        "n_steps": 7168,
        "gamma": 0.916,
        "learning_rate": 5.7e-5,
        "clip_range": 0.39,
        "gae_lambda": 0.98,
    }
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="mps",
        tensorboard_log=LOG_DIR,
        **model_params,
    )
    model.learn(total_timesteps=50_000, progress_bar=True)
    model.save("./opt/test_50k")
    env.close()
    print("done")

    


if __name__ == "__main__":
    main()






