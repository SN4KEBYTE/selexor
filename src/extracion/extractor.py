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

    @property
    def projection_matrix(self) -> np.ndarray:
        return self._w
