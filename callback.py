import os

import numpy as np

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, model_save_freq, model_save_path,
                 model_plot_freq, model_plot_path,
                 ep_len, model_eval_freq, eval_env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.model_eval_freq = model_eval_freq
        self.model_plot_freq = model_plot_freq
        self.model_plot_path = model_plot_path
        self.ep_len = ep_len
        self.eval_env = eval_env
        self.plotting = False
        self.plot_time = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.model.env.reset()

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.model_save_freq == 0:
            save_name = os.path.join(self.model_save_path, str(self.num_timesteps).zfill(10))
            self.save_model(save_name)
            print(f'Saving model {save_name}')
        if self.num_timesteps % self.model_eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
            print(f'Validation: {mean_reward}/{std_reward}')
            self.logger.record('Validation Reward', mean_reward)
        if self.num_timesteps % self.model_plot_freq == 1:
            self.trajectory_data = np.zeros((self.ep_len+1, 12), dtype=np.float32)
            self.plotting = True
            self.plot_time = 0
        if self.plotting:
            price_data = self.model.env.env_method('get_price_state', False, False)[0]
            account_data = self.model.env.env_method('get_account_state', False)[0]
            self.trajectory_data[self.plot_time, :5] = price_data[:, 1]
            self.trajectory_data[self.plot_time, 5:-1] = account_data
            self.trajectory_data[self.plot_time, -1] = self.model.env.get_attr('portfolio')[0]
            self.plot_time += 1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if self.plotting:
            prices = self.trajectory_data[:, :5]
            plt.figure()
            plt.subplot(511)
            plt.plot(prices[:, 0][:-1])                
            plt.subplot(512)
            plt.plot(prices[:, 1][:-1])
            plt.subplot(513)
            plt.plot(prices[:, 2][:-1])
            plt.subplot(514)
            plt.plot(prices[:, 3][:-1])
            plt.subplot(515)
            plt.plot(prices[:, 4][:-1])
            plt.savefig(os.path.join(self.model_plot_path, f'prices_{self.num_timesteps}.png'))

            accounts = self.trajectory_data[:, 5:]
            plt.figure()
            plt.subplot(711)
            plt.plot(accounts[:, 6][:-1])                
            plt.subplot(712)
            plt.plot(accounts[:, 0][:-1])
            plt.subplot(713)
            plt.plot(accounts[:, 1][:-1])
            plt.subplot(714)
            plt.plot(accounts[:, 2][:-1])
            plt.subplot(715)
            plt.plot(accounts[:, 3][:-1])
            plt.subplot(716)
            plt.plot(accounts[:, 4][:-1])
            plt.subplot(717)
            plt.plot(accounts[:, 5][:-1])             
            plt.savefig(os.path.join(self.model_plot_path, f'account_{self.num_timesteps}.png'))

            plt.close('all')
            self.plotting = False

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def save_model(self, name):
        self.model.save(name)