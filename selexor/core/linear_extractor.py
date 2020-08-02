from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.core.base_extractor import BaseExtractor


class LinearExtractor(BaseExtractor, ABC):
    def __init__(self, n_components: int) -> None:
        super(LinearExtractor, self).__init__(n_components)
        self._w: NDArray[Number] or None = None
        self._variance_explained: List or None = None

    @abstractmethod
    def fit(self):
        pass

    def transform(self, x_train: NDArray[Number]) -> NDArray[Number]:
        if self._w is None:
            raise RuntimeError('Projection matrix is not calculated. Please use fit method first, or fit_transform.')

        return x_train.dot(self._w)

    @abstractmethod
    def fit_transform(self) -> NDArray[Number]:
        pass

    def _calculate_projection_matrix(self, eigen_vals: NDArray[Number], eigen_vecs: NDArray[Number]) -> None:
        eigen_pairs: List[Tuple[Number, NDArray[Number]]] = sorted(
            [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))],
            key=lambda k: k[0], reverse=True)

        new_features: List[NDArray[Number]] = [eigen_pairs[i][1][:, np.newaxis].real for i in range(self._n_components)]
        self._w = np.hstack(new_features)

    def _calculate_variance_explained(self, eigen_vals: NDArray[Number]) -> None:
        eigen_sum = np.sum(eigen_vals)
        self._variance_explained = [val / eigen_sum for val in eigen_vals]

    @property
    def projection_matrix(self) -> NDArray[Number]:
        return self._w

    @property
    def variance_explained(self) -> NDArray[Number]:
        return self._variance_explained
