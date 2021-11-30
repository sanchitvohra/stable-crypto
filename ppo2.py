import gym
import numpy as np
import time

import torch
import torch.nn as nn

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO
from environment import CryptoEnv

from feature_extractor import CustomCombinedExtractor

from callback import CustomCallback

n_steps = 0
starting_balance = 1000000.0        # starting portfolio amount in dollars
max_trade = 10000.0                 # max number of $ amount for buy/sell
trading_fee = 0.01                  # trading fee during buy
history = 3                         # number of stacks in state

data = np.load("data/crypto_data.npy")
data_mean = np.mean(data, axis=1)
data_std = np.std(data, axis=1)

ep_len = 2048
eval_ep_len = 10000

env = CryptoEnv(data, starting_balance, max_trade, trading_fee, history, data_mean, data_std, ep_len, False)
eval_env = Monitor(CryptoEnv(data, starting_balance, max_trade, trading_fee, history, data_mean, data_std, eval_ep_len, True))

model_save_path = "logs/"
model_plot_path = "plots/"
model_save_freq = ep_len * 10
model_eval_freq = ep_len * 10
model_plot_freq = ep_len * 10

policy_kwargs = dict(
    # features_extractor_class=CustomCombinedExtractor,
    # features_extractor_kwargs=dict(features_dim=64),
    net_arch=[256, 256, dict(pi=[128, 64], vf=[128, 64])],
    activation_fn=nn.SiLU
)

# model = PPO('MultiInputPolicy', env, n_steps=ep_len, verbose=1, policy_kwargs=policy_kwargs,tensorboard_log="logs/", learning_rate= 1e-9)
model = PPO('MlpPolicy', env, verbose=1, n_steps=ep_len, tensorboard_log="logs/")
model.learn(1e7, callback=CustomCallback(model_eval_freq, model_save_path, model_plot_freq, model_plot_path, ep_len, model_eval_freq, eval_env))
