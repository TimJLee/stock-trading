


# environment란 agent 가 배울수 있는 기능들을 내포한 환경이다.

import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import TempVecEnv
from stable_baselines import PPO2
from env.TradingEnv import TradingEnv
import pandas as pd

# 파일을 읽어서 날짜 별로 sort 해준다.
df = pd.read_csv('./data/data.csv')
df = df.sort_values('Date')

# 알고리즘이 실행될수 있도록 벡터화 된 공간을 만들어 준다.
env = TempVecEnv([lambda: TradingEnv(df)])

# stable_baselines의 함수들을 불러온다
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=24451)

# 학습 반복횟수를 정해준다
obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()