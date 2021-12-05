import sys, os
import coloredlogs,logging
import yaml
import uuid
from utils import defaults
import numpy as np

import environment

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_env_wrapper(data, starting_balance, maximum_action, trading_fee, history, 
                     reward_scaling, episode_length, validation):
    def _init():
        env = Monitor(environment.CryptoEnv(data, starting_balance, maximum_action, trading_fee, 
                        history, reward_scaling, episode_length, validation))
        return env
    return _init

def setup_env(config):
    logger = logging.getLogger("common")
    if not os.path.exists("data/crypto_data.npy"):
        logger.error("No data file found! Ensure data/crypto_data.npy present")
        exit()

    data = np.load("data/crypto_data.npy")
    if "environment" not in config:
        logger.warn("No environment in config. Using default values")
    env_config = config.get("environment", {})
    starting_balance = env_config.get("starting-balance", defaults.ENV_STARTING_BALANCE)
    maximum_action = env_config.get("maximum-action", defaults.ENV_MAX_ACTION)
    trading_fee = env_config.get("trading-fee", defaults.ENV_TRADING_FEE)
    history = env_config.get("history", defaults.ENV_HISTORY)
    reward_scaling = env_config.get("reward-scaling", defaults.ENV_REWARD_SCALING)
    episode_length = env_config.get("episode-length", defaults.ENV_EPISODE_LENGTH)
    vectorized = env_config.get("vectorized", defaults.ENV_VECTORIZED)
    vectorized_size = env_config.get("vectorized-size", defaults.ENV_VECTORIZED_LENGTH)

    logger.info("Setting up environment:")
    logger.info(f'Starting balance: {starting_balance}')
    logger.info(f'Maximum action: {maximum_action}')
    logger.info(f'Trading fee: {trading_fee}')
    logger.info(f'History: {history}')
    logger.info(f'Reward Scaling: {reward_scaling}')
    logger.info(f'Episode Length: {episode_length}')
    logger.info(f'Vectorized: {vectorized}')
    if vectorized:
        logger.info(f'Vectorized Size: {vectorized_size}')

    if vectorized:
        env = SubprocVecEnv([make_env_wrapper(data, starting_balance, maximum_action, 
        trading_fee, history, reward_scaling, episode_length, False) 
        for i in range(vectorized_size)])
    else:
        env = make_env_wrapper(data, starting_balance, maximum_action, 
        trading_fee, history, reward_scaling, episode_length, False)() 

    validation_env = make_env_wrapper(data, starting_balance, maximum_action, 
        trading_fee, history, reward_scaling, episode_length, True)() 
    
    return env, validation_env

def main():
    logger = logging.getLogger("common")
    logger.setLevel(logging.INFO)
    coloredlogs.install(level=logging.INFO, logger=logger)

    if len(sys.argv) != 2:
        logger.error("No config file!")
        logger.error("Usage: python3 main.py <config.yml>")
        exit()

    config = sys.argv[1]

    if not os.path.exists(config):
        logger.error("Config file path does not exist!")
        logger.error("Usage: python3 main.py <config.yml>")
        exit()    

    try:
        config = open(config, 'r')
    except:
        logger.error("Unable to open config file!")
        exit() 

    try:
        config = yaml.safe_load(config)
    except:
        logger.error("Unable to parse config file!")
        exit()

    if "name" not in config:
        logger.info("No name in config")
        name = uuid.uuid1().hex
        logger.info(f'Generated name: {name}')
    else:
        name = config["name"]

    base_path = os.path.join('logs', name)

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_settings = config.get("logging", None)
    if log_settings:
        file_logging = config.get("logging", None)
        if file_logging:
            log_path = os.path.join(base_path, 'train.log')
            logger.info(f'Setting up logging to {log_path}')
            fileHandler = logging.FileHandler(log_path, mode='w')
            format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fileHandler.setFormatter(format)
            logger = logging.getLogger("common")
            logger.addHandler(fileHandler)

    env, val_env = setup_env(config)

    env.reset()
    for i in range(5):
        state, reward, _, _ = env.step(np.array([1, 0, 0, 0, 0], dtype=np.float32))
        print(state['buy-history'][0, :, :5], reward)
    for i in range(10):
        env.step(np.zeros(5, dtype=np.float32))
    for i in range(5):
        state, reward, _, _ = env.step(np.array([-1, 0, 0, 0, 0], dtype=np.float32))
        print(state['buy-history'][0, :, :5], reward)

if __name__ == "__main__":
    main()