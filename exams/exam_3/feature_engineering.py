import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils import split_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

class FeatureEngineering:

    def __init__(self, X, y, feature_selection_methods, vif_threshold, n_features):
        self.X = X
        self.y = y
        self.feature_selection_methods = feature_selection_methods
        self.vif_threshold = vif_threshold
        self.n_features = n_features

    def vif_scores(self):

        xs = MinMaxScaler().fit_transform(self.X)
        self.vif_df = pd.DataFrame({
            "Features": self.X.columns,
            "VIF Factor": [variance_inflation_factor(xs, i) for i in range(xs.shape[1])],
        })

        print("VIF scores:")
        print(self.vif_df.to_string(index=False))

    def vif(self, vif_threshold):
        self.vif_scores()
        selected = self.vif_df[self.vif_df["VIF Factor"] <= vif_threshold]
        feature_names = selected["Features"].tolist()
        return feature_names
    
    def rfe(self, n_features):
        selector = RFE(LogisticRegression(), n_features_to_select=n_features, step=1)
        xs = MinMaxScaler().fit_transform(self.X)
        selector.fit(xs, self.y)
        feature_names = self.X.columns[selector.support_].tolist()
        return feature_names

    def random_forest(self, n_features):

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        selector = SelectFromModel(
            estimator=rf,
            threshold=-np.inf,
            max_features=n_features
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
        if method_name == "VIF":
            return self.vif(vif_threshold=self.vif_threshold)
        elif method_name == "RFE":
            return self.rfe(n_features=self.n_features)
        elif method_name == "random_forest":
            return self.random_forest(n_features=self.n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method_name}")

    def main(self):
        feature_set = set()
        for feature_selection_method in self.feature_selection_methods:
            print(f"\nRunning feature selection method: {feature_selection_method}")
            feature_names = self.select_features(feature_selection_method)
            print(f"selected features in method {feature_selection_method}: {feature_names}")
            feature_set.update(set(feature_names))           

        print(f"feature set: {feature_set}")
        return list(feature_set)
    

