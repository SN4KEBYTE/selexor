from typing import Callable, Dict, List

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray
from scipy.spatial.distance import pdist, squareform


class KernelPCA:
    def __init__(self, k: int, gamma: float, kernel: str = 'rbf') -> None:
        self.__k: int = k
        self.__gamma: gamma = gamma
        self.__kernel_func: Callable = self.__get_kernel(kernel)
        self.__alphas: NDArray[Number] or None = None
        self.__lambdas: List or None = None

    def fit(self, x_train: NDArray[Number]) -> 'KernelPCA':
        mat_dists: NDArray[Number] = squareform(pdist(x_train, 'sqeuclidean'))

        k: NDArray[Number] = self.__kernel_func(mat_dists)

        n: int = k.shape[0]
        ones_mat: NDArray[Number] = np.ones((n, n)) / n
        k: NDArray[Number] = k - ones_mat.dot(k) - k.dot(ones_mat) + ones_mat.dot(k).dot(ones_mat)

        eigen_vals, eigen_vecs = np.linalg.eigh(k)

        self.__alphas = np.column_stack((eigen_vecs[:, -i]) for i in range(1, self.__k + 1))
        self.__lambdas = [eigen_vals[-i] for i in range(1, self.__k + 1)]

        return self

    def __project_point(self, point: NDArray[Number], x_train: NDArray[Number]) -> NDArray[Number]:
        dist: NDArray[Number] = np.array([np.sum(point - row) ** 2 for row in x_train])
        k: NDArray[Number] = self.__kernel_func(dist)

        return k.dot(self.__alphas / self.__lambdas)

    def transform(self, x: NDArray[Number], x_train: NDArray[Number]) -> NDArray[Number]:
        x_transformed = [self.__project_point(row, x_train) for row in x]

        return np.row_stack(x_transformed)

    def fit_transform(self, x: NDArray[Number], x_train: NDArray[Number]) -> NDArray[Number]:
        self.fit(x_train)

        return self.transform(x, x_train)

    def __get_kernel(self, kernel: str) -> Callable:
        kernels: Dict = {'rbf': self.__rbf}

        return kernels[kernel]

    def __rbf(self, mat_dists: NDArray[Number]) -> NDArray[Number]:
        return np.exp(-self.__gamma * mat_dists)

    @property
    def alphas(self):
        return self.__alphas

    @property
    def lambdas(self):
        return self.__lambdas
