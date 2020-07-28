from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Extractor(ABC):
    def __init__(self, k: int) -> None:
        self._k: int = k
        self._w: np.ndarray = np.array()
        self._variance_explained: List = []

    @abstractmethod
    def fit(self):
        pass

    def transform(self, x_train: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise RuntimeError('Projection matrix is not calculated. Please use fit method first, or fit_transform.')

        return x_train.dot(self._w)

    @abstractmethod
    def fit_transform(self) -> np.ndarray:
        pass

    def _calculate_projection_matrix(self, eigen_vals: np.ndarray, eigen_vecs: np.ndarray) -> None:
        eigen_pairs = sorted([(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))],
                             key=lambda k: k[0], reverse=True)

        new_features = [eigen_pairs[i][1][:, np.newaxis].real for i in range(self._k)]
        self._w = np.hstack(new_features)

    def _calculate_variance_explained(self, eigen_vals: np.ndarray) -> None:
        eigen_sum = np.sum(eigen_vals)
        self._variance_explained = [val / eigen_sum for val in eigen_vals]

    @property
    def projection_matrix(self) -> np.ndarray:
        return self._w

    @property
    def variance_explained(self):
        return self._variance_explained
