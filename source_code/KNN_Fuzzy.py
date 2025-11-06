from fuzzy_knn import *
import numpy as np

class FuzzyKNN:
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.knn = KNN(self.k, self.m) 
        self.fitted = False
        self.predicted = False

    def fit(self, x_train, y_train):
        if isinstance(x_train, np.ndarray):
            self.x_train = x_train.astype(np.float64)
            self.y_train = y_train.astype(np.int32)
        else:
            self.x_train = np.array(x_train, dtype = np.float64)
            self.y_train = np.array(y_train, dtype = np.int32)
        self.knn._fit(self.x_train, self.y_train) 
        self.fitted = True

    def predict(self, x_test):
        assert self.fitted, "Please call fit() before predicting"

        if not isinstance(x_test, np.ndarray):
            x_test = np.array([x_test])

        result = self.knn._predict(x_test)

        self.predictions = result.predictions
        self.memberships = result.memberships
        self.predicted = True
        return self.predictions

    def get_memberships(self):
        assert self.predicted, "Please call predict() before acess fuzzy membership"
        return self.memberships
