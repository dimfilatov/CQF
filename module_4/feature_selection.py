import pandas as pd
from sklearn.feature_selection import RFECV, RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

class FeatureSelection:
    """Boston housing price feature selection using VIF, SelectKBest, RFE, RFECV and SHAP."""

    def __init__(self, data):
        self.df = data
        self.pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        self.selected_features = {}

    def load_data(self):
        self.df = pd.read_csv(self.df)
        self.df.columns = self.df.columns.str.upper()
        return self.df

    def prepare_data(self, label):

        self.X = self.df.drop(columns=[label])
        self.y = self.df[label]

    def vif_scores(self):

        xs = StandardScaler().fit_transform(self.X)
        self.vif_df = pd.DataFrame({
            "Features": self.X.columns,
            "VIF Factor": [variance_inflation_factor(xs, i) for i in range(xs.shape[1])],
        })

        print("VIF scores:")
        print(self.vif_df.to_string(index=False))

    def method_vif(self, vif_threshold):
        self.vif_scores()
        selected = self.vif_df[self.vif_df["VIF Factor"] <= vif_threshold]
        columns = selected["Features"].tolist()
        self.X_subset = self.X[columns]
        self.selected_features["VIF"] = columns

    def method_select_k_best(self, k):
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        columns = self.X.columns[selector.get_support()].tolist()
        self.X_subset = self.X[columns]
        self.selected_features["SelectKBest"] = columns

    def method_rfe(self, n_features):
        selector = RFE(LinearRegression(), n_features_to_select=n_features, step=1)
        xs = StandardScaler().fit_transform(self.X)
        selector.fit(xs, self.y)
        columns = self.X.columns[selector.support_].tolist()
        self.selected_features["RFE"] = columns
        self.X_subset = self.X[columns]

    def method_rfecv(self, cv, scoring): 
        selector = RFECV(
            estimator=LinearRegression(),
            step=1,
            cv=cv,
            scoring=scoring,
        )
        xs = StandardScaler().fit_transform(self.X)
        selector.fit(xs, self.y)
        columns = self.X.columns[selector.support_].tolist()
        self.selected_features["RFECV"] = columns
        self.X_subset = self.X[columns]

    def fit_and_score(self, X_subset, method_name):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        model.fit(X_subset, self.y)
        score = model.score(X_subset, self.y)
        return {"method": method_name, "r2": score, "model": model}

    # def explain_shap(self, X_subset):
    #     model = Pipeline(
    #         steps=[
    #             ("scaler", StandardScaler()),
    #             ("regressor", LinearRegression()),
    #         ]
    #     )
    #     model.fit(X_subset, self.y)
    #     explainer = shap.Explainer(model.named_steps["regressor"], X_subset)
    #     shap_values = explainer(X_subset)
    #     return shap_values

    def select_features(self, method_name, **kwargs):
        if method_name == "VIF":
            return self.method_vif(vif_threshold=kwargs.get("vif_threshold", 5))
        elif method_name == "SelectKBest":
            return self.method_select_k_best(k=kwargs.get("k", 6))
        elif method_name == "RFE":
            return self.method_rfe(n_features=kwargs.get("n_features", 6))
        elif method_name == "RFECV":
            return self.method_rfecv(cv=kwargs.get("cv", 5), scoring=kwargs.get("scoring", "r2"))
        else:
            raise ValueError(f"Unknown feature selection method: {method_name}")

    def run(self, feature_selection_method):
        res=self.fit_and_score(self.X_subset, feature_selection_method)

        comparison = pd.DataFrame(
            [
                {"method": feature_selection_method, "r2": res["r2"], "selected_features": self.selected_features[feature_selection_method]}
            ]
        )

        return comparison

def main():
    df_path = "./data/boston.csv"
    label = "MEDV"
    feature_selection_methods = ["VIF", "SelectKBest", "RFE", "RFECV"]
    vif_threshold=5

    fs = FeatureSelection(data=df_path)
    fs.load_data()
    fs.prepare_data(label=label)
    for feature_selection_method in feature_selection_methods:
        print(f"\nRunning feature selection method: {feature_selection_method}")
        fs.select_features(feature_selection_method, vif_threshold=vif_threshold, k=6, n_features=6, cv=5, scoring="r2")
        results = fs.run(feature_selection_method=feature_selection_method)
        print(results.to_string(index=False))

if __name__ == "__main__":
    main()
