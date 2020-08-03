from typing import List

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray
from scipy.spatial.distance import pdist, squareform

from selexor.core.extractors.kernel_extractor import KernelExtractor


class KernelPCA(KernelExtractor):
    def __init__(self, n_components: int, gamma: float, kernel: str = 'rbf') -> None:
        super(KernelPCA, self).__init__(n_components, gamma, kernel)
        self.__alphas: NDArray[Number] or None = None
        self.__lambdas: List or None = None

    def fit(self, x_train: NDArray[Number]) -> 'KernelPCA':
        mat_dists: NDArray[Number] = squareform(pdist(x_train, 'sqeuclidean'))

        k: NDArray[Number] = self._kernel_func(mat_dists)

        n: int = k.shape[0]
        ones_mat: NDArray[Number] = np.ones((n, n)) / n
        k: NDArray[Number] = k - ones_mat.dot(k) - k.dot(ones_mat) + ones_mat.dot(k).dot(ones_mat)

        eigen_vals, eigen_vecs = np.linalg.eigh(k)

        self.__alphas = np.column_stack((eigen_vecs[:, -i]) for i in range(1, self._n_components + 1))
        self.__lambdas = [eigen_vals[-i] for i in range(1, self._n_components + 1)]

        return self

    def __project_point(self, point: NDArray[Number], x_train: NDArray[Number]) -> NDArray[Number]:
        dist: NDArray[Number] = np.array([np.sum(point - row) ** 2 for row in x_train])
        k: NDArray[Number] = self._kernel_func(dist)

        return k.dot(self.__alphas / self.__lambdas)

    def transform(self, x: NDArray[Number], x_train: NDArray[Number]) -> NDArray[Number]:
        x_transformed = [self.__project_point(row, x_train) for row in x]

        return np.row_stack(x_transformed)

    def fit_transform(self, x: NDArray[Number], x_train: NDArray[Number]) -> NDArray[Number]:
        self.fit(x_train)

        return self.transform(x, x_train)

    @property
    def alphas(self):
        return self.__alphas

    @property
    def lambdas(self):
        return self.__lambdas
