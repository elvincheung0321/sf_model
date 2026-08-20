import imageio.v2 as imageio
import numpy as np
from model import StreetFighter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
class StreetFighterWatch(StreetFighter):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        self.last_rgb = np.asarray(obs).copy()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        if info["health"] == 0 and info["enemy_health"] == 0:
            reward = 0
            self.enemy_health = 0
            self.player_health = 0
        else:
            dmg_dealt = self.enemy_health - info["enemy_health"]
            dmg_taken = self.player_health - info["health"]
            self.enemy_health = info["enemy_health"]
            self.player_health = info["health"]
            reward = dmg_dealt - dmg_taken
        return obs, reward, terminated, truncated, info
    def reset(self, seed=None, options=None):
        super(StreetFighter, self).reset(seed=seed)
        obs, info = self.game.reset(seed=seed, options=options)
        self.last_rgb = np.asarray(obs).copy()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        info = self.game.data.lookup_all()
        self.player_health = info.get("health", 0)
        self.enemy_health = info.get("enemy_health", 0)
        return obs, info

model_path = "./opt/checkpoints/sf_65m_v4_8700000_steps.zip"

env = DummyVecEnv([StreetFighterWatch])
env = VecFrameStack(env, n_stack=4, channels_order="last")
model = PPO.load(model_path, env=env)

speed = 8

obs = env.reset()
frames = []
while True:
    action, _ = model.predict(obs, deterministic=False)
    obs, rewards, dones, infos = env.step(action)
    frames.append(env.venv.envs[0].last_rgb)
    if dones[0]:
        break
env.close()
imageio.mimsave("sf_65m_v4_8.7m{}x.mp4".format(speed), frames, fps=30 * speed)
print("saved mp4")
