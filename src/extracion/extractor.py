import numpy as np
from abc import ABC, abstractmethod


class Extractor(ABC):
    def __init__(self, k: int):
        self.__k: int = k
        self.__w = None

    @abstractmethod
    def fit(self):
        pass

    def transform(self, x_train: np.ndarray) -> np.ndarray:
        if self.__w is None:
            raise RuntimeError('Projection matrix is not calculated. Please use fit method first, or fit_transform.')

        return x_train.dot(self.__w)

    @abstractmethod
    def fit_transform(self):
        pass

    @property
    def projection_matrix(self):
        return self.__w
