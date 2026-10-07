import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []

    def DGP(self, n_samples, beta):
        self.feature = np.random.normal(0, 1, size=n_samples).reshape(-1, 1)
        self.unit_vector = np.ones((n_samples, 1))
        self.X = np.hstack((self.unit_vector, self.feature))
        print(f"X shape: {self.X.shape}, beta shape: {beta.shape}")
        self.error = np.random.normal(0, 1, size=n_samples)
        self.y = beta.T @ self.X.T + self.error

    def fit(self):
        self.beta_hat = np.zeros((self.X.shape[1], 1))
        self.loss_history = []
        for _ in range(self.n_iterations):
            predictions = self.beta_hat.T @ self.X.T
            errors = predictions - self.y
            print(f"Errors shape: {errors.shape}, X shape: {self.X.shape}, beta_hat shape: {self.beta_hat.shape}")
            gradient = -2 * errors @ self.X / self.X.shape[0]
            print(f"Gradient shape: {gradient.shape}, beta_hat shape: {self.beta_hat.shape}")
            self.beta_hat = self.beta_hat + self.learning_rate * gradient.T
            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)
        print(f"Estimated coefficients: {self.beta_hat.flatten()}")

    def plot_loss(self):
        plt.plot(range(self.n_iterations), self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.show()

if __name__ == "__main__":
    learning_rate = 0.01
    n_iterations = 1000
    beta = np.array([[2], [1]])
    n_samples = 1000

    gd_model = GradientDescent(learning_rate, n_iterations)
    gd_model.DGP(n_samples, beta)
    gd_model.fit()
    gd_model.plot_loss()