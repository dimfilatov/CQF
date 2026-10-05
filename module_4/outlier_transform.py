from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OutlierTransform(BaseEstimator, TransformerMixin):
    """Clip values to lower and upper percentile bounds."""

    def __init__(self, q_lower=5, q_upper=95):
        self.q_lower = q_lower
        self.q_upper = q_upper

    def fit(self, X):
        X = np.asarray(X)
        self.lower_ = np.percentile(X, self.q_lower, axis=0)
        self.upper_ = np.percentile(X, self.q_upper, axis=0)

    def transform(self, X):
        X = np.asarray(X).copy()
        idx_lower = X < self.lower_
        idx_upper = X > self.upper_
        for i in range(X.shape[1]):
            X[idx_lower[:, i], i] = self.lower_[i]
            X[idx_upper[:, i], i] = self.upper_[i]
        return X

    def run(self, path_to_data, periods):
        """Example from the PDF showing outlier clipping on market return data."""
        spx = pd.read_csv(path_to_data, index_col=0, parse_dates=True)["2015":]
        rdict = {f"{period}D_RET": spx["Adj Close"].pct_change(period) for period in periods}
        rdf = pd.DataFrame(rdict).dropna()
        X = rdf.values

        self.fit(X)
        Xt = self.transform(X)
        return rdf, X, Xt

    @staticmethod
    def plot_histograms(X, Xt):
        """Plot original vs transformed 1-day return distribution."""
        _, bins, _ = plt.hist(
            X[:, 0],
            density=True,
            bins=200,
            alpha=1,
            color='b',
            label='Original',
        )
        plt.hist(
            Xt[:, 0],
            density=True,
            bins=bins,
            alpha=1,
            color='r',
            label='Transformed',
        )
        plt.title('Original vs Transformed Distribution')
        plt.xlim(-0.05, 0.05)
        plt.legend()
        plt.show()

def main():
    spx_path = "./data/SPY.csv"
    periods = [1, 5, 20, 60, 120]

    outlier_transform = OutlierTransform(q_lower=5, q_upper=95)
    rdf, X, Xt = outlier_transform.run(spx_path, periods=periods)
    print("\nSP500 return summary:")
    print(rdf.describe().to_string())
    print("\nOriginal vs transformed sample shape:", X.shape, Xt.shape)

    outlier_transform.plot_histograms(X, Xt)


if __name__ == "__main__":
    main()