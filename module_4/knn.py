import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances

import io
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score


class KNN:
    def __init__(self, N, mu_vector, sigma):
        self.N = N
        self.mu_vector = mu_vector
        self.sigma = sigma
        self.seed = 0

    def generate_data(self):
        self.X, self.y = make_blobs(n_samples=30, 
                        centers=[[self.mu_vector[0], self.mu_vector[0]], 
                                   [self.mu_vector[1], self.mu_vector[1]]], 
                        cluster_std=self.sigma, 
                        random_state=110)
        self.df = pd.DataFrame({'X1': self.X[:,0],
                            'X2': self.X[:,1],
                            'y': self.y})
        self.df['Class'] = self.df['y'].map({0: 'Blue', 1: 'Red'})

    def compute_euclidean_distance(self, x1, x2):
        self.df['Euclidean'] = euclidean_distances(self.df[['X1', 'X2']], np.array([[x1, x2]]))

    def classify_knn(self, x1, x2, k):
        self.compute_euclidean_distance(x1, x2)
        self.df = self.df.sort_values(by='Euclidean')
        self.df['Rank'] = range(1, len(self.df) + 1)
        self.df['KNN_Class'] = self.df['Class'].where(self.df['Rank'] <= k)
        print(f"Classifying point ({x1}, {x2}) using {k} nearest neighbours:")
        print(self.df[['X1', 'X2', 'y', 'Euclidean', 'Rank', 'KNN_Class']].where(self.df['Rank'] <= k).dropna())
        
    def plot_boundary(self, k, step):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.X, self.y)

        # Plot the decision boundary.
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        # Create Meshgrid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        # Predict labels for each point in mesh
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        # Reshape to match dimensions
        Z = Z.reshape(xx.shape)
        # Plotting
        plt.contour(xx, yy, Z, cmap=plt.cm.bwr, linestyles = 'dashed', linewidths=0.5)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.bwr)
        plt.title(f'KNN Decision Boundary with {k} Nearest Neighbours')
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$', rotation='horizontal')
        plt.show()
   
if __name__ == "__main__":
    # Example usage
    knn_model = KNN(N=30, mu_vector=[-0.5, 0.5], sigma=0.4)
    knn_model.generate_data()
    knn_model.classify_knn(x1=0.25, x2=0.25, k=5)
    knn_model.plot_boundary(k=5, step=0.1)

    knn_model.load_data("./binary.csv")
    knn_model.run_pipeline()