import os
from typing import Union

import implicit
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.linalg import svd
from scipy import sparse


class ALSModel:
    def __init__(self, r, alpha=40, lmbda=50, num_recommendations=10, num_factors=100):
        self.lmbda = lmbda
        self.users, self.items = r.shape
        # Confidence matrix
        self.c = 1 + alpha * r

        # Preference matrix
        self.p = r

        # User latent factor matrix
        self.x = np.random.rand(self.users, num_factors).astype(np.float32) * 0.01

        # Item latent factor matrix
        self.y = np.random.rand(self.items, num_factors).astype(np.float32) * 0.01

        self.num_recommendations = num_recommendations

    def least_squares_step(self, X, Y, optimize=Union["user", "item"]):
        if optimize == "user":
            YtY = Y.T @ Y
            for i in range(X.shape[0]):
                Cu = np.diag(self.c[i])
                A = YtY + Y.T @ (Cu - np.identity(Y.shape[0])) @ Y + self.lmbda * np.identity(Y.shape[1])
                b = Y.T @ Cu @ self.p[i]
                X[i] = np.linalg.solve(A, b)
        elif optimize == "item":
            XtX = X.T @ X
            for j in range(Y.shape[0]):
                Ci = np.diag(self.c[:, j])
                A = XtX + X.T @ (Ci - np.identity(X.shape[0])) @ X + self.lmbda * np.identity(X.shape[1])
                b = X.T @ Ci @ self.p[:, j]
                Y[j] = np.linalg.solve(A, b)

    def loss(self):
        # Calculate loss to make sure it's decreasing.
        total = 0

        item_scores = self.x @ self.y.T
        confidence_weighted = self.c * (self.p - item_scores) ** 2
        total += np.sum(confidence_weighted)
        total += self.lmbda * (np.sum(self.x**2) + np.sum(self.y**2))

        return total / (self.x.shape[0] * self.y.shape[0])

    def backward(self):
        # Update user latent factor matrix

        self.least_squares_step(self.x, self.y, optimize="user")
        self.least_squares_step(self.x, self.y, optimize="item")

    def forward(self, X):
        return X @ self.y.T
