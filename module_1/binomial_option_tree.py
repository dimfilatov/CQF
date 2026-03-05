import numpy as np
from numpy import zeros, maximum, power, sqrt, around

class binomial_tree:
    """Binomial tree option pricing model.
    
    Implements a binomial tree to calculate option prices and Greeks (delta)
    for both European and American style options (calls and puts).
    Uses a multiplicative binomial model with up and down movements.
    """

    def __init__(self, 
                 steps: int,
                 t: int,
                 r: float,
                 sig: float,
                 spot: float,
                 strike: float,
                 option_type: str,
                 option_style: str):
        """Initialize binomial tree option pricing model parameters.
        
        Args:
            steps (int): Number of steps in the binomial tree.
            t (int): Time to maturity in years.
            r (float): Risk-free interest rate.
            sig (float): Volatility (annualized).
            spot (float): Current spot price of underlying asset.
            strike (float): Strike price of the option.
            option_type (str): Type of option ('call' or 'put').
            option_style (str): Style of option ('european' or 'american').
        """
        self.steps = steps
        self.t = t
        self.r = r
        self.sig = sig
        self.spot = spot
        self.strike = strike
        self.option_type = option_type
        self.option_style = option_style
        self.dt = self.t/self.steps
        self.u = 1 + self.sig * np.sqrt(self.dt)
        self.d = 1 - self.sig * np.sqrt(self.dt)
        self.p_u = 0.5 + self.r * np.sqrt(self.dt) / (2 * self.sig)
        self.df = 1 / (1 + self.r * self.dt)
    
    def binomial_option(self) -> tuple:
        """Calculate option price and Greeks using binomial tree model.
        
        Constructs a binomial tree by forward-looping through asset prices
        and computing intrinsic values, then backward-loops to calculate
        option values using risk-neutral probabilities and delta hedging ratios.
        
        Returns:
            tuple: Four numpy arrays containing:
                - spot: Asset prices at each node
                - payoff: Intrinsic payoff values at maturity
                - value: Option values at each node
                - delta: Hedging ratios at each node
        """
        
        # create placeholders for asset prices, option payoff, option value and delta
        spot = zeros((self.steps + 1, self.steps + 1))
        payoff = zeros((self.steps + 1, self.steps + 1))
        value = zeros((self.steps + 1, self.steps + 1))
        delta = zeros((self.steps + 1, self.steps + 1))
        
        # forward loop
        for i in range(self.steps + 1):
            for j in range(i+1):
                spot[j, i] = self.spot * self.u ** (i - j) * self.d ** j
                if self.option_type == 'call':
                    payoff[j, i] = max(spot[j, i] - self.strike, 0) 
                if self.option_type == 'put':
                    payoff[j, i] = max(self.strike - spot[j, i], 0)
        
        # backward loop
        for i in range(self.steps, -1, -1):
            for j in range(i + 1):
                if i == self.steps:
                    value[j, i] = payoff[j, i]
                    delta[j, i] = 0
                else:
                    if self.option_style == 'european':
                        value[j, i] = self.df * (self.p_u * value[j, i + 1] + (1 - self.p_u) * value[j + 1, i + 1])
                    if self.option_style == 'american': 
                        value[j, i] = max(self.df * (self.p_u * value[j, i + 1] + (1 - self.p_u) * value[j + 1, i + 1]), payoff[j, i])
                    delta[j, i] = (value[j, i + 1] - value[j + 1, i + 1]) / (spot[j, i + 1] - spot[j + 1, i + 1])
        return around(spot,2), around(payoff, 2), around(value, 2), around(delta, 4)

if __name__ == "__main__":
    # Example usage
    steps = 4
    t = 1
    r = 0.05
    sig = 0.2
    spot = 100
    strike = 100
    option_type = 'call'
    option_style = 'european'
    
    tree = binomial_tree(steps, t, r, sig, spot, strike, option_type, option_style)
    spot, payoff, value, delta = tree.binomial_option()
    
    print("Spot Prices:\n", spot)
    print("Payoff at Maturity:\n", payoff)
    print("Option Values:\n", value)
    print("Delta Values:\n", delta)
        
        
    