import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils import split_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
import matplotlib.pyplot as plt

class FeatureEngineering:

    def __init__(self, X, y, feature_selection_methods, corr_threshold, n_features):
        self.X = X
        self.y = y
        self.feature_selection_methods = feature_selection_methods
        self.corr_threshold = corr_threshold
        self.n_features = n_features
    
    def correlation_analysis(self):

        corr = self.X.corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        features_to_drop = [
            col for col in upper.columns
            if any(upper[col] > self.corr_threshold)
        ]

        return features_to_drop

    def plot_correlation_heatmap(self):

        self.corr = self.X.corr()
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(
            self.corr,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1
        )

        # Axis labels
        ax.set_xticks(np.arange(len(self.corr.columns)))
        ax.set_yticks(np.arange(len(self.corr.index)))

        ax.set_xticklabels(self.corr.columns, rotation=90)
        ax.set_yticklabels(self.corr.index)

        # Display correlation coefficients
        for i in range(len(self.corr)):
            for j in range(len(self.corr.columns)):
                ax.text(
                    j, i,
                    f"{self.corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

        # Colorbar
        fig.colorbar(im, ax=ax, label="Correlation")

        ax.set_title("Feature Correlation Matrix")

        plt.tight_layout()
        plt.show()

    def method_rfe(self): 
        selector = RFE(
            estimator=LogisticRegression(),
            step=1,
            n_features_to_select=self.n_features
        )
        xs = StandardScaler().fit_transform(self.X)
        selector.fit(xs, self.y)
        feature_names = self.X.columns[selector.support_].tolist()
        return feature_names

    def random_forest(self):

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        selector = SelectFromModel(
            estimator=rf,
            threshold=-np.inf,
            max_features=self.n_features
        )
        
        selector.fit(self.X, self.y)

        feature_names = self.X.columns[
            selector.get_support()
        ]
        importance = pd.DataFrame({
                            "feature": self.X.columns,
                            "importance": selector.estimator_.feature_importances_,
                            "selected": selector.get_support()
                        })

        display(
            importance[importance["selected"]]
            .sort_values("importance", ascending=False)
        )
        return feature_names

    def select_features(self, method_name):
        if method_name == "RFE":
            return self.method_rfe()
        elif method_name == "random_forest":
            return self.random_forest()
        else:
            raise ValueError(f"Unknown feature selection method: {method_name}")

    def main(self):
        feature_set = set()
        feature_d = dict()
        features_to_drop = self.correlation_analysis()
        print(f"features to drop due to correlation analysis: {features_to_drop}, with length: {len(features_to_drop)}")
        self.X = self.X.drop(columns=features_to_drop)
        for i, feature_selection_method in enumerate(self.feature_selection_methods):
            print(f"\nRunning feature selection method: {feature_selection_method}")
            feature_names = self.select_features(feature_selection_method)
            feature_d[feature_selection_method] = feature_names
            print(f"selected features in method {feature_selection_method}: {feature_names}")
            if i == 0:
                feature_set.update(set(feature_names))           
            else:
                feature_set &= set(feature_names)

        feature_list = sorted(feature_set)
        self.X = self.X[feature_list]
        self.plot_correlation_heatmap()

        print(f"feature set: {feature_set}")

        feature_df = pd.DataFrame({
            k: pd.Series(v) for k, v in feature_d.items()
        })

        return feature_list, self.corr,feature_df
    

