from typing import List, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

from selexor.core.extractors.kernel_extractor import KernelExtractor


class KernelPCA(KernelExtractor):
    """
    Kernel principal component analysis extractor.
    """

    def __init__(self, n_components: int, gamma: float, kernel: str = 'rbf') -> None:
        """
        Initialize the class with some values.

        :param n_components: desired dimension of the new feature space.
        :param gamma: kernel coefficient for RBF.
        :param kernel: kernel type.

        :return: None.
        """

        super(KernelPCA, self).__init__(n_components, gamma, kernel)
        self.__x_train: Optional[np.ndarray] = None
        self.__alphas: Optional[np.ndarray] = None
        self.__lambdas: Optional[List] = None

    def fit(self, x: np.ndarray) -> 'KernelPCA':
        """
        A method that fits the dataset in order to extract features.

        :param x: samples. This dataset will be remembered, because KernelPCA uses training samples to apply
                  dimensionality reduction to the new samples.

        :return: fitted extractor.
        """

        self.__x_train = x

        mat_dists: np.ndarray = squareform(pdist(x, 'sqeuclidean'))

        k: np.ndarray = self._kernel_func(mat_dists)

        n: int = k.shape[0]
        ones_mat: np.ndarray = np.ones((n, n)) / n
        k: np.ndarray = k - ones_mat.dot(k) - k.dot(ones_mat) + ones_mat.dot(k).dot(ones_mat)

        eigen_vals, eigen_vecs = np.linalg.eigh(k)

        self.__alphas = np.column_stack((eigen_vecs[:, -i]) for i in range(1, self._n_components + 1))
        self.__lambdas = [eigen_vals[-i] for i in range(1, self._n_components + 1)]

        return self

    def __project_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        A method that projects one sample to the new feature space.

        :param sample: sample to be projected.

        :return: projected sample.
        """

        dist: np.ndarray = np.array([np.sum(sample - row) ** 2 for row in self.__x_train])
        k: np.ndarray = self._kernel_func(dist)

        return k.dot(self.__alphas / self.__lambdas)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        A method that applies dimensionality reduction to a given samples.

        :param x: samples.

        :return: samples projected onto a new space.
        """

        x_transformed = [self.__project_sample(row) for row in x]

        return np.row_stack(x_transformed)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        A method that fits the dataset and applies dimensionality reduction to a given samples.

        :param x: samples.

        :return: samples projected onto a new space.
        """

        self.fit(x)

        return self.transform(x)

    @property
    def alphas(self):
        """
        Eigen vectors corresponding to the biggest n_components eigen values.

        :return: eigen vectors or None in case fit (or fit_transform) was not called.
        """

        return self.__alphas

    @property
    def lambdas(self):
        """
        The biggest n_components eigen values.

        :return: eigen values or None in case fit (or fit_transform) was not called.
        """

        return self.__lambdas
