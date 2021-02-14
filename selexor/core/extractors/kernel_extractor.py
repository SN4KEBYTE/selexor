from abc import ABC, abstractmethod
from typing import Dict, Callable

import numpy as np

from selexor.core.base.base import Base


class KernelExtractor(Base, ABC):
    """
    Abstract base class for all extractors which use kernel trick.
    """

    def __init__(self, n_components: int, gamma: float = 1.0, kernel: str = 'rbf') -> None:
        """
        Initialize the class with some values.

        :param n_components: desired dimension of the new feature space.
        :param gamma: kernel coefficient for RBF.
        :param kernel: kernel type.

        :return: None.
        """

        super(KernelExtractor, self).__init__(n_components)
        self._gamma: float = gamma
        self._kernel_func: Callable = self.__get_kernel(kernel)

    @abstractmethod
    def fit(self, x: np.ndarray) -> 'KernelExtractor':
        """
        A method that fits the dataset in order to extract features. This is an abstract method and
        must be implemented in subclasses.

        :param x: samples.

        :return: fitted extractor.
        """

        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        A method that applies dimensionality reduction to a given samples. This is an abstract method and
        must be implemented in subclasses.

        :param x: samples.

        :return: samples projected onto a new space.
        """

        pass

    @abstractmethod
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        A method that fits the dataset and applies dimensionality reduction to a given samples. This is an abstract
        method and must be implemented in subclasses.

        :param x: samples.

        :return: samples projected onto a new space.
        """

        pass

    def __get_kernel(self, kernel: str) -> Callable:
        """
        A method that sets the kernel function.

        :param kernel: kernel type.

        :return: kernel function.
        """

        kernels: Dict[str, Callable] = {'rbf': self.__rbf}

        return kernels[kernel]

    def __rbf(self, mat_dists: np.ndarray) -> np.ndarray:
        """
        Gaussian radial basis function.

        :param mat_dists: distances matrix.

        :return: Gaussian RBF values.
        """

        return np.exp(-self._gamma * mat_dists)
