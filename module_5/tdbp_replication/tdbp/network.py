"""The TDBP model in TensorFlow (slides 10-11, 19-21).

Time:  one network per grid step, P_0, ..., P_{N-1}.
Space: P_i maps the state at t_i to beta(t_i) E[ g / beta(T) | F(t_i) ].

Training runs backwards in time with temporal-difference targets:
    U_N      = g(S_N)
    target_i = e^{-r dt} * p_i * U_{i+1}                 (p_i: barrier survival)
    P_i      = argmin  E_w[(P_i(x_i) - target_i)^2]
    U_i      = (1 - lambda) P_i(x_i) + lambda * target_i  (TD(lambda))
One set of weights is trained step after step (each network starts from the
next step's weights) and a copy is kept for every step.
"""
import numpy as np
import tensorflow as tf

LEAKY = 0.01
TF_ACT = {"leaky_relu": lambda x: tf.nn.leaky_relu(x, alpha=LEAKY), "relu": tf.nn.relu,
          "tanh": tf.tanh, "softplus": tf.nn.softplus}
NP_ACT = {"leaky_relu": lambda x: np.where(x > 0, x, LEAKY * x), "relu": lambda x: np.maximum(x, 0.0),
          "tanh": np.tanh, "softplus": lambda x: np.logaddexp(0.0, x)}


def learning_rate(schedule, lr, step, total):
    """Schedules of slide 19 / Figure 5 (Optax formulas), as TensorFlow ops.
    'exponential': flat for 10%, then lr * 0.01^progress (floored at 1e-15)
    'cosine'     : lr * ((1 - a) (0.5 (1 + cos(pi t/T)))^3 + a), a = 1e-15
    'exp_cosine' : exponential decay to lr/2 on the first half, then cosine to ~0
    'constant'   : lr"""
    step = tf.cast(step, tf.float32)
    total = tf.maximum(tf.cast(total, tf.float32), 1.0)

    def exponential(lr0, t, T, rate):
        begin = tf.floor(0.1 * T)
        progress = tf.maximum(t - begin, 0.0) / tf.maximum(T - begin, 1.0)
        return tf.maximum(lr0 * tf.pow(rate, progress), 1e-15)

    def cosine(lr0, t, T, power):
        frac = tf.minimum(t, T) / tf.maximum(T, 1.0)
        return lr0 * ((1 - 1e-15) * tf.pow(0.5 * (1 + tf.cos(np.pi * frac)), power) + 1e-15)

    if schedule == "constant":
        return lr * tf.ones_like(step)
    if schedule == "exponential":
        return exponential(lr, step, total, 0.01)
    if schedule == "cosine":
        return cosine(lr, step, total, 3.0)
    if schedule == "exp_cosine":
        half = tf.maximum(tf.floor(total / 2), 1.0)
        return tf.where(step < half, exponential(lr, step, half, 0.5),
                        cosine(0.5 * lr, step - half, total - half, 1.0))
    raise ValueError(f"unknown schedule {schedule!r}")


class Optimizer:
    """Adam, AMSGrad, AdaBelief or SGD with resettable state (slide 19: best
    convergence with Adam, AdaBelief, AMSGrad). Written directly in TensorFlow
    because AdaBelief is not in core TensorFlow and the state must be reset for
    every time step without rebuilding the compiled training graph."""

    def __init__(self, name, schedule, lr, params):
        self.name, self.schedule, self.lr, self.params = name, schedule, lr, params
        self.b1, self.b2 = 0.9, 0.999
        self.eps = 1e-8
        self.step = tf.Variable(0.0, trainable=False)
        self.total = tf.Variable(1.0, trainable=False)
        self.m = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.v = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]

    def reset(self, total_steps):
        self.step.assign(0.0)
        self.total.assign(float(max(total_steps, 1)))
        for v in self.m + self.v:
            v.assign(tf.zeros_like(v))

    def apply(self, grads):
        lr = learning_rate(self.schedule, self.lr, self.step, self.total)
        self.step.assign_add(1.0)
        t, b1, b2 = self.step, self.b1, self.b2
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m.assign(b1 * m + (1 - b1) * g)
            v.assign(b2 * v + (1 - b2) * tf.square(g))
            m_hat = m / (1 - tf.pow(b1, t))
            v_hat = v / (1 - tf.pow(b2, t))
            p.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + self.eps))


class TDBPModel:
    """feature_fn(paths, step) -> (n_paths, n_features) network inputs.
    price_scale divides the targets (the strike) so the networks fit O(1) values."""

    def __init__(self, n_features, payoff, n_steps, dt, r, price_scale, depth, hidden,
                 activation, optimizer, schedule, lr, batch_size, epochs, warm_start_epochs,
                 td_lambda, hard_knockout, seed):
        self.payoff = payoff
        self.n_steps, self.dt, self.r, self.scale = n_steps, dt, r, price_scale
        self.activation, self.batch_size = activation, batch_size
        self.epochs, self.warm_start_epochs = epochs, warm_start_epochs
        self.td_lambda, self.hard_knockout, self.seed = td_lambda, hard_knockout, seed
        self.sizes = [n_features] + [hidden] * depth + [1]
        self.params = []
        for m, n in zip(self.sizes[:-1], self.sizes[1:]):
            self.params += [tf.Variable(tf.zeros((m, n))), tf.Variable(tf.zeros(n))]
        self.optimizer = Optimizer(optimizer, schedule, lr, self.params)
        self.rng = np.random.default_rng(seed)
        self.steps = {}                    # step -> (weights, x_mean, x_std)
        self._train_epoch = tf.function(self._epoch)

    def _init_weights(self):
        """He-normal weights, zero biases."""
        rng = np.random.default_rng(self.seed)
        for k, (m, n) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            self.params[2 * k].assign((rng.standard_normal((m, n)) * np.sqrt(2.0 / m)).astype(np.float32))
            self.params[2 * k + 1].assign(np.zeros(n, dtype=np.float32))


    def _forward(self, x):
        act, n_layers = TF_ACT[self.activation], len(self.params) // 2
        for k in range(n_layers):
            x = tf.matmul(x, self.params[2 * k]) + self.params[2 * k + 1]
            if k < n_layers - 1:
                x = act(x)
        return x[:, 0]

    def _epoch(self, X, y, batches):
        """One epoch over the mini-batches (compiled once, reused for every step)."""
        total = tf.constant(0.0)
        for b in tf.range(tf.shape(batches)[0]):
            idx = batches[b]
            xb, yb = tf.gather(X, idx), tf.gather(y, idx)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(tf.square(self._forward(xb) - yb)) / tf.cast(tf.shape(xb)[0], tf.float32)
            self.optimizer.apply(tape.gradient(loss, self.params))
            total += loss
        return total / tf.cast(tf.shape(batches)[0], tf.float32)

    def _fit_network(self, X, y, epochs):
        n = len(X)
        bs = min(self.batch_size, n)
        n_batches = max(n // bs, 1)
        X, y = (tf.constant(a, tf.float32) for a in (X, y))
        self.optimizer.reset(epochs * n_batches)
        for _ in range(epochs):
            batches = self.rng.permutation(n)[: n_batches * bs].reshape(n_batches, bs)
            loss = float(self._train_epoch(X, y, tf.constant(batches, tf.int32)))
        return loss

    def _network_value(self, step, X):
        """Numpy forward pass with the weights stored for ``step``."""
        weights, mean, std = self.steps[step]
        h = (X - mean) / std
        n_layers = len(weights) // 2
        for k in range(n_layers):
            h = h @ weights[2 * k] + weights[2 * k + 1]
            if k < n_layers - 1:
                h = NP_ACT[self.activation](h)
        return h[:, 0] * self.scale

    def fit(self, paths, verbose):
        """Train on paths starting at step 0; ``weights`` = sample weights or None."""
        S = paths["S"]
        alive, survival = self.payoff.alive(S), self.payoff.survival(S)
        disc = np.exp(-self.r * self.dt)
        U = self.payoff.terminal(S[:, -1])
        self._init_weights()
        for i in range(self.n_steps - 1, -1, -1):
            target = disc * survival[:, i] * U
            X = np.log(paths["S"][:, [i]] / self.scale)
            mean, std = X.mean(0), X.std(0)
            std[std < 1e-8] = 1.0
            epochs = self.epochs if i == self.n_steps - 1 else self.warm_start_epochs
            loss = self._fit_network((X - mean) / std, target / self.scale, epochs)
            self.steps[i] = ([p.numpy().copy() for p in self.params], mean, std)
            V = self._network_value(i, X)
            if self.hard_knockout:
                V = V * alive[:, i]
            U = (1 - self.td_lambda) * V + self.td_lambda * target
            if verbose and (i % max(self.n_steps // 10, 1) == 0 or i == self.n_steps - 1):
                print(f"  [TDBP] step {i:4d}/{self.n_steps}  loss {loss:.3e}  mean value {V.mean():.4f}")
        return self

    def predict(self, paths, step):
        """Value on every path at ``step``; knocked-out barrier paths are worth 0 (slide 30)."""
        S = paths["S"]
        alive = self.payoff.alive(S)[:, step] if self.hard_knockout else 1.0
        if step == self.n_steps:
            return self.payoff.terminal(S[:, step]) * alive
        return self._network_value(step, np.log(paths["S"][:, [step]] / self.scale)) * alive
