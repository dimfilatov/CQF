import numpy as np
import pandas as pd
# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class GridSearch:

    def __init__(self, cv, penalty, C):
        self.cv = cv
        self.penalty = penalty
        self.C = C

    def load_data(self, data_path, label):
        df = pd.read_csv(data_path)
        self.features = df.drop(label, axis=1)
        self.target = df[label]

        self.X = self.features.values
        self.y = self.target.values

    def build_pipeline(self):
        self.pipe = Pipeline([("scaler", StandardScaler()),
                        ("logistic", LogisticRegression(solver='liblinear'))])

    def create_param_grid(self):
        self.param_grid = dict(logistic__C=self.C, logistic__penalty=self.penalty)

    def run_pipeline(self):
        self.build_pipeline()
        self.create_param_grid()
        grid = GridSearchCV(self.pipe, self.param_grid, cv=self.cv, n_jobs=-1, verbose=1)
        best_model = grid.fit(self.X, self.y)
        print(f"Best Penalty: {best_model.best_params_['logistic__penalty']}")
        print(f"Best C: {best_model.best_params_['logistic__C']}")
        print(f"Best Score: {best_model.best_score_:.04}") 


if __name__ == "__main__":
    cv=5
    penalty=['l1', 'l2']
    C=np.linspace(0.01, 10, 10)
    data_path = "./data/binary.csv"
    label = "admit"
    grid_search = GridSearch(cv=cv, penalty=penalty, C=C)
    grid_search.load_data(data_path, label)
    grid_search.run_pipeline()
    