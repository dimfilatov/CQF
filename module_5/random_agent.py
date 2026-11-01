import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class observation_space:
    def __init__(self, n):
        self.shape = (n,)

class action_space:
    def __init__(self, n):
        self.n = n
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        return random.random()

class Investing:
    def __init__(self, S0, T, r_, mu_, sigma_, steps, amount):
        self.initial_value = S0
        self.maturity = T
        self.short_rate_ = r_
        self.index_drift_ = mu_
        self.volatility_ = sigma_
        self.steps = steps
        self.initial_balance = amount
        self.portfolio_value = amount
        self.portfolio_value_new = amount
        self.observation_space = observation_space(4)
        self.osn = self.observation_space.shape[0]
        self.action_space = action_space(1)
        self._generate_data()
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _generate_data(self):
        s = [self.initial_value]
        self.short_rate = random.choice(self.short_rate_)
        self.index_drift = random.choice(self.index_drift_)
        self.volatility = random.choice(self.volatility_)
        self.dt = self.maturity / self.steps
        for t in range(1, self.steps + 1):
            st = s[t - 1] * math.exp(((self.index_drift -
            self.volatility ** 2 / 2) * self.dt +
            self.volatility * math.sqrt(
            self.dt) * random.gauss(0, 1))
            )
            s.append(st)
        self.data = pd.DataFrame(s, columns=['Xt'])
        self.data['Yt'] = self.initial_value * np.exp(
        self.short_rate * np.arange(len(self.data)) * self.dt)

    def _get_state(self):
        Xt = self.data['Xt'].iloc[self.bar]
        Yt = self.data['Yt'].iloc[self.bar]
        return np.array([Xt, Yt, self.xt, self.yt]), {}
    
    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
        
    def reset(self):
        self.bar = 0
        self.xt = 0
        self.yt = 0
        self.treward = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_value_new = self.initial_balance
        self.episode += 1
        self._generate_data()
        self.state, _ = self._get_state()
        return self.state, _

    def add_results(self, pl):
        df = pd.DataFrame({'e': self.episode, 'xt': self.xt,
        'yt': self.yt, 'pv': self.portfolio_value,
        'pv_new': self.portfolio_value_new, 'p&l[$]': pl,
        'p&l[%]': pl / self.portfolio_value_new,
        'Xt': self.state[0], 'Yt': self.state[1],
        'Xt_new': self.new_state[0],
        'Yt_new': self.new_state[1],
        'r': self.short_rate, 'mu': self.index_drift,
        'sigma': self.volatility}, index=[0])
        self.portfolios = pd.concat((self.portfolios, df),ignore_index=True)

    def step(self, action):
        self.bar += 1
        self.new_state, _ = self._get_state()
        if self.bar == 1:
            self.xt = action
            self.yt = (1 - action)
            pl = 0.
            reward = 0.
            self.add_results(pl)
        else:
            self.portfolio_value_new = (
                            self.xt * self.portfolio_value * self.new_state[0] / self.state[0] +
                            self.yt * self.portfolio_value * self.new_state[1] / self.state[1])
            pl = self.portfolio_value_new - self.portfolio_value
            self.xt = action
            self.yt = (1 - action)
            self.add_results(pl)
            reward = pl
            self.portfolio_value = self.portfolio_value_new
        if self.bar == len(self.data) - 1:
            done = True
        else:
            done = False
        self.state = self.new_state
        return self.state, reward, done, False, {}

if __name__ == "__main__":
    S0 = 1
    investing = Investing(S0=S0, 
                        T=1.0, 
                        r_=[0.05], 
                        mu_=[0.3],
                        sigma_=[0.35], 
                        steps=252, 
                        amount=1)
    investing.seed(750)
    investing._generate_data()
    investing.data.plot(style=['g--', 'b:'], lw=1.0)
    plt.xlabel('time step')
    plt.ylabel('price')
    plt.show()
    investing.reset()
    for _ in range(investing.steps - 1):
        investing.step(investing.action_space.sample())

    investing.portfolios[['Xt', 'Yt', 'pv']].plot(
                                                title='PORTFOLIO VALUE | RANDOM AGENT',
                                                style=['g--', 'b:', 'r-'], lw=1)
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.show()