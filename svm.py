import numpy as np
import cvxopt as cx # quadratic programming software

class svm:
    def __init_(self, kernel="polynomial", degree=2, C=1):
        self.kernel = kernel
        if self.kernel == "polynomial":
            self.degree = degree
        self.C = C

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass