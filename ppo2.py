import numpy as np
import torch.nn as nn
import sys, os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable

from stable_baselines3 import PPO
from environment import CryptoEnv

from callback import CustomCallback

n_steps = 0
starting_balance = 1000000.0        # starting portfolio amount in dollars
max_trade = 10000.0                 # max number of $ amount for buy/sell
trading_fee = float(sys.argv[2])                   # trading fee during buy
history = int(sys.argv[3])                         # number of stacks in state
reward_scaling = float(sys.argv[4])

data = np.load("data/crypto_data.npy")
data_mean = np.mean(data, axis=1)
data_std = np.std(data, axis=1)

ep_len = int(sys.argv[5])
eval_ep_len = 10000

def make_env(eval):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Monitor(CryptoEnv(data, starting_balance, max_trade, trading_fee, history, reward_scaling, data_mean, data_std, ep_len, eval))
        return env
    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def exponential_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return (progress_remaining ** 2) * initial_value

    return func

if __name__ == "__main__":
    test_name = sys.argv[1]
    test_dir_path = os.path.join("logs/", test_name)

    os.mkdir(test_dir_path)

    model_save_path = os.path.join(test_dir_path, "checkpoints")
    model_plot_path = os.path.join(test_dir_path, "plots")
    model_save_freq = 25 # every n rollouts
    model_eval_freq = 10 # every n rollouts
    model_plot_freq = 25 # every n rollouts

    os.mkdir(model_save_path)
    os.mkdir(model_plot_path)

    num_cpu = 1
    env = SubprocVecEnv([make_env(False) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_env(True) for i in range(num_cpu)])
    # env = CryptoEnv(data, starting_balance, max_trade, trading_fee, history, reward_scaling, data_mean, data_std, ep_len, False)
    # eval_env = Monitor(CryptoEnv(data, starting_balance, max_trade, trading_fee, history, reward_scaling, data_mean, data_std, eval_ep_len, True))

    policy_kwargs = dict(
        # features_extractor_class=CustomCombinedExtractor,
        # features_extractor_kwargs=dict(features_dim=64),
        net_arch=[2048, 2048, 1024, 512, 512, dict(pi=[256, 256], vf=[256, 256])],
    )

    # model = PPO('MultiInputPolicy', env, n_steps=ep_len, verbose=1, policy_kwargs=policy_kwargs,tensorboard_log="logs/", learning_rate= 1e-9)
    model = PPO(
                'MlpPolicy', 
                env, 
                verbose=1,
                n_steps=ep_len,
                clip_range=exponential_schedule(0.2), 
                batch_size=1024,
                # clip_range_vf=1e-5,
                # target_kl=5e-6,
                learning_rate=exponential_schedule(float(sys.argv[6])),
                policy_kwargs=policy_kwargs,
                tensorboard_log=test_dir_path
            )

    model.learn(
        num_cpu * ep_len * 120, 
        tb_log_name="tb_logs",
        callback=CustomCallback(model_save_freq, model_save_path, model_plot_freq, model_plot_path, ep_len, model_eval_freq, eval_env)
        )

## kl: 5e-6