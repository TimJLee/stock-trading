import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

Max_Account_Balance = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 10000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 30000
INITIAL_ACCOUNT_BALANCE = 10000


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, Max_Account_Balance)

        # 행동
        # 사고 x%, 팔고 x%, 가지고 있고, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 가격
        # 캔들 차트의 시가, 종가, 고가, 저가를 다음 5개의 가격에 대해 포함
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def reset(self):
        # 환경 state를 초기화 해준다
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # 현재 스텝을 랜덤 포인트로 데이터 프레임 내에서 변경해준다.
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)
        return self.observation_next()




    def observation_next(self):
        # 지난 5일간의 주식 정보를 가지고 와서 0과 1 사이로 스케일 해준다.
        # AKA 타임스텝을 에이전트의 계정 정보 와 더하고 스케일 해준다.
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # 추가 데이터 와 스케일을 0 과 1 사이에서 추가 해준다
        obs = np.append(frame, [[
            self.balance / Max_Account_Balance,
            self.max_net_worth / Max_Account_Balance,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs




    def action_take(self, action):
        # 타임스텝 내부에서 현재 가격을 랜덤하게 돌려준다
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    # 각 스텝 별로 특정한 action(model에 의해 선택된)을 하고 reward를 계산하고 다음 관측을 리턴한다
    def step(self, action):
        self.action_take(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self.observation_next()

        return obs, reward, done, {}


    def render(self, mode='human', close=False):
        # 환경이 어떻게 진행되는지 프린트 해준다
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE


        print(f'스텝: {self.current_step}')
        print(f'잔고: {self.balance}')
        print(
            f'가지고 있는 주식: {self.shares_held} (총 매도량: {self.total_shares_sold})')
        print(
            f'평균 매수 금액: {self.cost_basis} (총 판매액: {self.total_sales_value})')
        print(
            f'순가치: {self.net_worth} (최대 순 가치: {self.max_net_worth})')
        print(f'이익: {profit}')
        print()