from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.core.base.base import Base


class LinearExtractor(Base, ABC):
    def __init__(self, n_components: int) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired dimension of the new feature space.

        :return: None.
        """

        super(LinearExtractor, self).__init__(n_components)
        self._w: Optional[NDArray[Number]] = None
        self._variance_explained: Optional[List] = None

    @abstractmethod
    def fit(*args, **kwargs):
        """
        A method that fits the dataset in order to extract features.

        :param args: variable length argument list.
        :param kwargs: arbitrary keyword arguments.

        :return: fitted extractor.
        """

        pass

    def transform(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        A method that applies dimensionality reduction to a given samples.

        :param x: samples.

        :return: samples projected onto a new space.

        :raises: RuntimeError: thrown when projection matrix is not calculated. In this case you need to use fit method
                 first (or fit_transform).
        """

        if self._w is None:
            raise RuntimeError('Projection matrix is not calculated. Please use fit method first, or fit_transform.')

        return x.dot(self._w)

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> NDArray[Number]:
        """
        A method that fits the dataset and applies dimensionality reduction to a given samples. This is an abstract
        method and must be implemented in subclasses.

        :param args: variable length argument list.
        :param kwargs: arbitrary keyword arguments.

        :return: samples projected onto a new space.
        """

        pass

    def _calculate_projection_matrix(self, eigen_vals: NDArray[Number], eigen_vecs: NDArray[Number]) -> None:
        """
        A method that calculates projection matrix using eigen values and eigen vectors. Matrix is stored in _w
        argument.

        :param eigen_vals: eigen values.
        :param eigen_vecs: eigen vectors corresponding to eigen values.

        :return: None.
        """

        eigen_pairs: List[Tuple[Number, NDArray[Number]]] = sorted(
            [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))],
            key=lambda k: k[0], reverse=True)

        new_features: List[NDArray[Number]] = [eigen_pairs[i][1][:, np.newaxis].real for i in range(self._n_components)]
        self._w = np.hstack(new_features)

    def _calculate_variance_explained(self, eigen_vals: NDArray[Number]) -> None:
        """
        A method that calculates explained variance using eigen values. It is stored in _variance_explained argument.

        :param eigen_vals: eigen values.

        :return: None.
        """

        eigen_sum = np.sum(eigen_vals)
        self._variance_explained = [val / eigen_sum for val in eigen_vals]

    @property
    def projection_matrix(self) -> Optional[NDArray[Number]]:
        """
        Projection matrix.

        :return: projection matrix or None in case fit (or fit_transform) was not called.
        """

        return self._w

    @property
    def variance_explained(self) -> Optional[List]:
        """
        Explained variance.

        :return: explained variance or None in case fit (or fit_transform) was not called.
        """

        return self._variance_explained
