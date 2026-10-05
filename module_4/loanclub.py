import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text

class LoanClub:
    def __init__(self, train, test, validation, label):
        self.train = train
        self.test = test
        self.validation = validation
        self.label = label

    def extract_features_and_labels(self):
        # remove target column to create feature only dataset
        self.X_train = self.train.drop(self.label, axis=1)
        self.X_val = self.validation.drop(self.label, axis=1)
        self.X_test = self.test.drop(self.label, axis=1)

        # store target column
        self.y_train = self.train[self.label]
        self.y_val = self.validation[self.label]
        self.y_test = self.test[self.label]

    def compute_entropy(self, y):
        prob_1 = len(y[y == 1]) / len(y)
        prob_2 = 1.0 - prob_1
        entropy = -prob_1 * np.log2(prob_1) - prob_2 * np.log2(prob_2)

        return entropy

    def entropy_gain(self, feature):

        initial_entropy = self.compute_entropy(y=self.y_train)
        print("Initial entropy of the target variable:", initial_entropy)

        # Calculate the new entropy of the feature values
        new_entropy = 0.0
        unique_values = self.X_train[feature].unique()
        len_train = len(self.y_train)
        for value in unique_values:
            subset_y = self.y_train[self.X_train[feature] == value]
            weight = len(subset_y) / len_train
            new_entropy += weight * self.compute_entropy(y=subset_y)

        # Calculate the information gain
        information_gain = initial_entropy - new_entropy
        print("New entropy of the feature values:", new_entropy)
        print("Information gain for feature", feature, "=", information_gain)

    def plot_decision_tree(self, clf, feature_names):
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=feature_names, filled=True)
        plt.show()

    def build_decision_tree(self, criterion, max_depth, min_samples_split, min_samples_leaf):

        clf = DecisionTreeClassifier(criterion=criterion, 
                                     max_depth=max_depth, 
                                     min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf, 
                                     random_state=0)
        clf = clf.fit(self.X_train, self.y_train)
        self.plot_decision_tree(clf, self.X_train.columns)

        train_score = clf.score(self.X_train, self.y_train)
        test_score = clf.score(self.X_test, self.y_test)

        print('train_score=', train_score)
        print('test_score=', test_score)



if __name__ == "__main__":
    train = pd.read_excel('data/lendingclubtraindata.xlsx')
    validation=pd.read_excel('data/lendingclubvaldata.xlsx')
    test=pd.read_excel('data/lendingclubtestdata.xlsx')
    feature = 'home_ownership'
    label = 'loan_status'
    criterion='entropy'
    max_depth=2
    min_samples_split=1000
    min_samples_leaf=200

    loan_club = LoanClub(train, test, validation, label, )
    loan_club.extract_features_and_labels()
    loan_club.entropy_gain(feature)
    loan_club.build_decision_tree(criterion, max_depth, min_samples_split, min_samples_leaf)
