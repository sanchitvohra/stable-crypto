import random
import gym
from gym import spaces
import numpy as np
import heapq

class CryptoEnv(gym.Env):
    def __init__(self, data, balance, max_trade, trading_fee, history_len, reward_scaling, ep_len, validate):
        super(CryptoEnv, self).__init__()

        # 'open', 'high', 'low', 'close', 'Volume USD', 'RSI', 'MACD', 'CCI', 'ADX'
        self.data = data
        self.trading_fee = trading_fee
        self.history_len = history_len
        self.starting_balance = balance
        self.max_trade = max_trade
        self.reward_scaling = reward_scaling
        self.eval = validate
        self.ep_len = ep_len

        self.coins = {'BCH': 0, 'BTC': 1, 'ETH': 2, 'LTC': 3, 'XRP': 4}
        self.num_coins = len(self.coins)

        self.compute_normalization()
        state = self.reset()
        self.action_space = spaces.Box(low=-1, high=1, shape=(5, ), dtype=np.float32)
        self.observation_space = spaces.Dict({'price' : spaces.Box(low=0, high=1, shape=state['price'].shape, dtype=np.float32),
                                            'account': spaces.Box(low=0, high=1, shape=state['account'].shape, dtype=np.float32),
                                            'buy-history': spaces.Box(low=0, high=1, shape=state['buy-history'].shape, dtype=np.float32)})

    def reset(self):
        if not self.eval:
            self.state_index = random.randint(0, self.data.shape[1] - 50000 - self.history_len)
        else:
            self.state_index = self.data.shape[1] - 50000 - self.history_len

        self.state = self.data[:, self.state_index, :]
        self.prev_states = []

        for i in range(self.history_len):
            self.prev_states.append(self.normalize_state(self.state))
            self.state_index += 1
            self.state = self.data[:, self.state_index, :]
        
        self.n_step = 0
        self.portfolio = self.starting_balance
        self.balance = self.portfolio
        self.account = np.zeros(len(self.coins), dtype=np.float32)
        self.account_dollars = np.zeros(len(self.coins), dtype=np.float32)
        self.buy_history = {}

        for coin in self.coins:
            self.buy_history[self.coins[coin]] = []

        state = self.get_state()
        return state

    def step(self, actions):
        exec_actions = actions * self.max_trade

        reward = 0
        for coin in self.coins.keys():
            if exec_actions[self.coins[coin]] < 0:
                reward += self.sell(self.coins[coin], -1 * exec_actions[self.coins[coin]])

        for coin in self.coins.keys():
            if exec_actions[self.coins[coin]] > 0:
                self.buy(self.coins[coin], exec_actions[self.coins[coin]])

        if self.history_len > 0:
            self.prev_states = self.prev_states[1:]
            self.prev_states.append(self.normalize_state(self.state))


        reward = reward * self.reward_scaling
        self.state_index += 1
        self.n_step += 1

        self.state = self.data[:, self.state_index, :] # move to new state
        self.account_dollars = self.state[:, 3] * self.account # update account dollars
        self.portfolio = self.balance + np.sum(self.account_dollars) 

        if self.portfolio <= 0.75 * self.starting_balance:
            return self.get_state(), -1 * self.starting_balance, True, {}
        elif not self.eval:
            return self.get_state(), reward, self.n_step == self.ep_len, {}
        else:
            return self.get_state(), reward, self.state_index == (self.data.shape[1] - 1), {}

    def buy(self, coin, amount):
        if amount > self.balance:
            amount = self.balance
            
        trading_fee = amount * self.trading_fee
        buy_amount = amount - trading_fee
        buy_quantity = buy_amount / self.state[coin, 3]

        self.account[coin] += buy_quantity
        self.balance -= amount

        heapq.heappush(self.buy_history[coin], (self.state[coin, 3], buy_quantity, self.n_step))

    def sell(self, coin, amount):
        quantity = amount / self.state[coin, 3]

        if quantity > self.account[coin]:
            quantity = self.account[coin]

        sell_amount = quantity * self.state[coin, 3]
        sell_quantity = quantity

        self.account[coin] -= sell_quantity
        self.balance += sell_amount

        reward = 0
        while(1):
            buy_price, buy_quantity, _ = heapq.heappop(self.buy_history[coin])
            reward += sell_quantity * (self.state[coin, 3] - buy_price)
            if buy_quantity > sell_quantity:
                heapq.heappush(self.buy_history[coin], (buy_price, buy_quantity - sell_quantity, self.n_step))
            else:
                break
        return reward

    def compute_normalization(self):
        self.max_norm = np.max(self.data, axis=1)
        self.min_norm = np.min(self.data, axis=1)

    def normalize_state(self, state):
        normalized_state = np.copy(state)
        normalized_state = (normalized_state - self.min_norm) / (self.max_norm - self.min_norm + 1e-12)
        return normalized_state

    def get_state(self):
        normalized_state = self.normalize_state(self.state)
        if self.history_len > 0:
            normalized_history = np.stack(self.prev_states)
        normalized_account = np.copy(self.account_dollars)
        normalized_account = normalized_account / (10 * self.starting_balance)
        normalized_balance = np.copy(self.balance)
        normalized_balance = normalized_balance / (10 * self.starting_balance)
        
        normalized_state = np.expand_dims(normalized_state, 0)
        price_state = np.vstack([normalized_history, normalized_state])
        account_state = np.hstack([normalized_balance, normalized_account]).reshape(-1)

        normalized_buy_history = np.zeros((5, 2, 20), dtype=np.float32)
        for coin in range(self.num_coins):
            coin_buy_history = self.buy_history[coin][:min(len(self.buy_history[coin]), 20)]
            coin_buy_history = list(zip(*coin_buy_history))
            if len(coin_buy_history) > 0:
                coin_price_history = np.array(list(coin_buy_history[0]))
                coin_quantity_history = np.array(list(coin_buy_history[1]))
                normalized_buy_history[coin, 0, :len(coin_price_history)] = (coin_price_history - self.min_norm[coin, 3]) / (self.max_norm[coin, 3] - self.min_norm[coin, 3] + 1e-12) 
                normalized_buy_history[coin, 1, :len(coin_quantity_history)] = coin_quantity_history / (self.max_trade / self.min_norm[coin, 3])
            
        return {'price': price_state, 'account': account_state, 'buy-history': normalized_buy_history}

    def get_price_state(self):
        price_state = np.copy(self.state[:, :5])
        return price_state

    def get_account_state(self):
        normalized_account = np.copy(self.account_dollars)
        normalized_balance = np.copy(self.balance)
        return np.hstack([normalized_balance, normalized_account])
