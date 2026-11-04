import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import A2C, PPO

PRICE_COLS = ["close", "open", "high", "low"]
ALGOS = {"PPO": PPO, "A2C": A2C}

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000, max_shares=100):
        super().__init__()
        self.df = df

        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)  # 0 = buy, 1 = sell all, 2 = hold
        # Test prices can exceed train maxima, so the upper bound is open.
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.current_step = 0
        self.total_steps = len(df)
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.max_shares = max_shares
        self.cumulative_rewards = []

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.df.iloc[self.current_step]["close"]
        self.cumulative_rewards = []
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array(
            [
                self.balance / self.initial_balance,
                self.shares_held / self.max_shares,
                self.current_price / self.df["close"].max(),
                row["open"] / self.df["open"].max(),
                row["high"] / self.df["high"].max(),
                row["low"] / self.df["low"].max(),
            ],
            dtype=np.float32,
        )

    def step(self, action):
        prev_portfolio_value = self.balance + self.shares_held * self.current_price

        # The decision is made on the observation at t, executed at close[t+1].
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
            self.balance += self.shares_held * self.current_price
            self.shares_held = 0

        new_portfolio_value = self.balance + self.shares_held * self.current_price
        reward = new_portfolio_value - prev_portfolio_value
        self.cumulative_rewards.append(reward)

        terminated = self.current_step >= self.total_steps - 1
        truncated = False
        return self._next_observation(), reward, terminated, truncated, {}


def train_agent(env, model_class, model_params, train_timesteps, seed, tensorboard_log=None):
    model = model_class(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        **model_params,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=train_timesteps, tb_log_name="Policy")
    return model


def test_agent(env, model):
    obs, _ = env.reset()
    for _ in range(len(env.df) - 1):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    return np.cumsum(env.cumulative_rewards)


def buy_and_hold(env):
    """Buy at close[1] (the first price the agent can trade at), hold to the end.
    Returns cumulative P&L aligned with the agent's cumulative rewards."""
    prices = env.df["close"].to_numpy()
    entry = prices[1]
    shares = min(env.initial_balance // entry, env.max_shares)
    cash = env.initial_balance - shares * entry
    values = cash + shares * prices[1:]
    return values - env.initial_balance


def max_drawdown(pnl, initial_balance):
    equity = initial_balance + np.asarray(pnl)
    peak = np.maximum.accumulate(equity)
    return ((equity - peak) / peak).min()


def plot_comparison(agent_curves, bh_pnl, name):
    curves = np.vstack(agent_curves)
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    x = np.arange(curves.shape[1])

    plt.figure(figsize=(10, 5))
    for c in curves:
        plt.plot(x, c, color="tab:blue", alpha=0.15, linewidth=0.8)
    plt.plot(x, mean, color="tab:blue", label=f"{name} mean ({len(curves)} seeds)")
    plt.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.2, label="±1 std")
    plt.plot(x, bh_pnl, color="black", linestyle="--", label="Buy & Hold")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative P&L")
    plt.title(f"Test Period: {name} vs Buy & Hold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nifty50.csv")
    parser.add_argument("--algo", default="A2C", choices=ALGOS.keys())
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    print("Data loaded successfully")

    ratio = 0.80
    train_size = int(df.shape[0] * ratio)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    print(f"Training Data: {train.shape}, Test Data: {test.shape}")

    test_env = StockTradingEnv(test)
    initial_balance = test_env.initial_balance

    bh = buy_and_hold(test_env)
    bh_dd = max_drawdown(bh, initial_balance)

    model_class = ALGOS[args.algo]
    params = {"learning_rate": args.lr}

    agent_curves = []
    for seed in range(args.seeds):
        print(f"Training {args.algo}, seed {seed}...")
        train_env = StockTradingEnv(train)  # fresh env per seed
        model = train_agent(
            train_env, model_class, params, args.timesteps, seed=seed
        )
        pnl = test_agent(test_env, model)
        agent_curves.append(pnl)
        print(
            f"  seed {seed}: final P&L = {pnl[-1]:,.0f}, "
            f"max DD = {max_drawdown(pnl, initial_balance):.2%}"
        )

    finals = np.array([c[-1] for c in agent_curves])
    dds = np.array([max_drawdown(c, initial_balance) for c in agent_curves])

    print("\n=== Test period summary ===")
    print(
        f"{args.algo} ({args.seeds} seeds): final P&L mean = {finals.mean():,.0f}, "
        f"std = {finals.std(ddof=1) if len(finals) > 1 else 0:,.0f}, "
        f"min = {finals.min():,.0f}, max = {finals.max():,.0f}; "
        f"mean max DD = {dds.mean():.2%}"
    )
    print(f"Buy & Hold: final P&L = {bh[-1]:,.0f}, max DD = {bh_dd:.2%}")
    print(f"Seeds beating Buy & Hold: {(finals > bh[-1]).sum()}/{len(finals)}")

    plot_comparison(agent_curves, bh, args.algo)


if __name__ == "__main__":
    main()