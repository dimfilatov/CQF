import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO, A2C


class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000, max_shares=100):
        super().__init__()
        self.df = df
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

        self.current_step = 0
        self.total_steps = len(df)
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.max_shares = max_shares
        self.cumulative_rewards = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.df.iloc[self.current_step]["close"]
        self.cumulative_rewards = []
        return self._next_observation(), {}

    def _next_observation(self):
        frame = np.array(
            [
                self.balance / self.initial_balance,
                self.shares_held / self.max_shares,
                self.current_price / self.df["close"].max(),
                self.df.iloc[self.current_step]["open"] / self.df["open"].max(),
                self.df.iloc[self.current_step]["high"] / self.df["high"].max(),
                self.df.iloc[self.current_step]["low"] / self.df["low"].max(),
            ],
            dtype=np.float32,
        )
        return frame

    def step(self, action):
        prev_portfolio_value = self.balance + self.shares_held * self.current_price

        self.current_step += 1
        self.current_price = self.df.iloc[self.current_step]["close"]

        if action == 0:
            shares_to_buy = min(
                self.balance // self.current_price,
                self.max_shares - self.shares_held,
            )
            self.balance -= shares_to_buy * self.current_price
            self.shares_held += shares_to_buy
        elif action == 1:
            shares_to_sell = self.shares_held
            self.balance += shares_to_sell * self.current_price
            self.shares_held = 0

        new_portfolio_value = self.balance + self.shares_held * self.current_price
        reward = new_portfolio_value - prev_portfolio_value
        self.cumulative_rewards.append(reward)

        terminated = self.current_step >= self.total_steps - 1
        truncated = False
        obs = self._next_observation()
        return obs, reward, terminated, truncated, {}


def train_agent(env, model_class, model_params, train_timesteps, tensorboard_log=None):
    model = model_class(
        policy="MlpPolicy",
        env=env,
        **model_params,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=train_timesteps, tb_log_name="Policy")
    return model


def test_agent(env, model):
    obs, _ = env.reset()
    for _ in range(len(env.df) - 1):
        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    return np.cumsum(env.cumulative_rewards)


def plot_cumulative_rewards(rewards, name):
    plt.plot(rewards, label=name)
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Cumulative Reward - {name}")
    plt.legend()
    plt.show()


# Main function
def main():
    # Get stock data
    df = pd.read_csv('data/nifty50.csv', index_col=0, parse_dates=True)
    print("Data loaded successfully")

    # Split the data into training and test sets
    ratio = 0.80
    train_size = int(df.shape[0]*ratio)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    print("Data split successfully")
    print(f"Traning Data: {train.shape}, Test Data: {test.shape}")

    # Initialize the training & test environment
    train_env = StockTradingEnv(train)
    test_env = StockTradingEnv(test)
    print("Environments initialized successfully")

    # Define RL algorithm
    algorithm = {
        "PPO": (PPO, {"learning_rate": 0.0003}),
        # "A2C": (A2C, {"learning_rate": 0.0003}),
    }

    # Train, test, and plot for each algorithm
    for name, (algo, params) in algorithm.items():
        print("Training the agent...")
        model = train_agent(train_env, algo, params, train_timesteps=100000)
        rewards_train = np.cumsum(train_env.cumulative_rewards)
        plot_cumulative_rewards(rewards_train, name)

        print("Testing the agent...")
        rewards = test_agent(test_env, model)

        # Plot the cumulative rewards for each algorithm
        plot_cumulative_rewards(rewards, name)

if __name__ == "__main__":
    main()