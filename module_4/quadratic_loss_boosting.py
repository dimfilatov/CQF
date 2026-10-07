from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class QuadraticLossBoosting:
    
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.trees = []

    def DGP(self, n_samples):

        self.X = np.random.normal(0, 1, size=n_samples).reshape(-1, 1)
        self.error = np.random.normal(0, 1, size=n_samples)
        self.y = 3 * self.X[:, 0]**2 + self.error

    def fit(self):
        # Fit the first tree
        tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_reg1.fit(self.X, self.y)
        self.trees.append(tree_reg1)
        self.plot()
        self.plot_tree(0)  # Plot the current tree after fitting

        # Fit subsequent trees on the residuals
        for i in range(1, self.n_estimators):
            residuals = self.y - self.predict(self.X)
            tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
            tree_reg.fit(self.X, residuals)
            self.trees.append(tree_reg)
            self.plot()
            self.plot_tree(i)  # Plot the current tree after fitting

    def predict(self, X):
        predictions = sum(tree.predict(X) for tree in self.trees)
        return predictions

    def plot_tree(self, tree_index):
        if tree_index < 0 or tree_index >= len(self.trees):
            raise ValueError("Invalid tree index.")
        tree = self.trees[tree_index]
        plt.figure(figsize=(12, 8))
        plot_tree(tree, filled=True)
        plt.title(f"Decision Tree {tree_index + 1}")
        plt.show()

    def plot(self):
        if not self.trees:
            raise ValueError("Fit the model before plotting.")

        x_range = np.linspace(self.X.min(), self.X.max(), 100).reshape(-1, 1)
        previous_trees = self.trees[:-1]
        previous_prediction = sum(
            tree.predict(self.X) for tree in previous_trees
        ) if previous_trees else np.zeros_like(self.y)
        previous_prediction_range = sum(
            tree.predict(x_range) for tree in previous_trees
        ) if previous_trees else np.zeros(len(x_range))
        tree_contribution = self.trees[-1].predict(self.X)
        tree_contribution_range = self.trees[-1].predict(x_range)
        prediction = previous_prediction + tree_contribution
        residual_before = self.y - previous_prediction
        residual_after = self.y - prediction
        stage = len(self.trees)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axes[0].scatter(self.X[:, 0], self.y, color='steelblue', alpha=0.45,
                        label='Observed y')
        axes[0].plot(x_range[:, 0], previous_prediction_range, '--',
                     color='gray', label='Previous prediction')
        axes[0].plot(x_range[:, 0], previous_prediction_range + tree_contribution_range,
                     color='crimson', label='New prediction = previous + tree')
        axes[0].set_ylabel('y / prediction')
        axes[0].set_title(
            f"Stage {stage}: $F_{stage}(x) = F_{stage - 1}(x) + T_{stage}(x)$"
        )
        axes[0].legend()

        axes[1].plot(x_range[:, 0], tree_contribution_range, color='darkorange')
        axes[1].set_ylabel(f'$T_{stage}(x)$')
        axes[1].set_title('Contribution from the newly added tree')

        axes[2].scatter(self.X[:, 0], residual_before, color='darkorange',
                        alpha=0.4, label=r'Before: $r=y-F_{m-1}(x)$')
        axes[2].scatter(self.X[:, 0], residual_after, color='seagreen',
                        alpha=0.4, label=r'After: $r=y-F_m(x)$')
        axes[2].axhline(0, color='black', linewidth=0.8)
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Residual')
        axes[2].set_title(
            r'Next tree is trained on the previous residual; '
            r'new residual = previous residual $-T_m(x)$'
        )
        axes[2].legend()

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    n_samples = 1000
    n_estimators = 3
    boosting_model = QuadraticLossBoosting(n_estimators=n_estimators)
    boosting_model.DGP(n_samples=n_samples)
    boosting_model.fit()
    predictions = boosting_model.predict(boosting_model.X)
    print(f"Predictions: {predictions[:5]}")  # Print first 5 predictions
