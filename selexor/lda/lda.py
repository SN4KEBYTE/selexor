from typing import List

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.core.extractors.linear_extractor import LinearExtractor


class LDA(LinearExtractor):
    """
    Linear discriminant analysis selector.
    """

    def __init__(self, n_components: int) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired dimension of the new feature space.

        :return: None
        """

        super(LDA, self).__init__(n_components)

    def fit(self, x: NDArray[Number], y: NDArray[Number]) -> 'LDA':
        """
        A method that fits the dataset in order to extract features.

        :param x: samples.
        :param y: class labels.

        :return: fitted extractor.
        """

        labels: NDArray[Number] = np.sort(np.unique(y))

        mean_vecs: List[NDArray[Number]] = [np.mean(x[y == label], axis=0) for label in labels]

        dim: int = x.shape[1]
        s_w: NDArray[Number] = np.zeros((dim, dim))

        for label, mean_vec in zip(labels, mean_vecs):
            class_scatter: NDArray[Number] = np.cov(x[y == label].T)
            s_w += class_scatter

        mean_overall: NDArray[Number] = np.mean(x, axis=0)
        s_b: NDArray[Number] = np.zeros((dim, dim))

        for i, mean_vec in enumerate(mean_vecs):
            n: int = x[y == i, :].shape[0]
            mean_vec: NDArray[Number] = mean_vec.reshape(dim, 1)
            mean_overall: NDArray[Number] = mean_overall.reshape(dim, 1)
            s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))

        self._calculate_explained_variance(eigen_vals)
        self._calculate_projection_matrix(eigen_vals, eigen_vecs)

        return self

    def fit_transform(self, x: NDArray[Number], y: NDArray[Number]) -> NDArray[Number]:
        """
        A method that fits the dataset and applies dimensionality reduction to a given samples.

        :param x: samples.
        :param y: class labels.

        :return: samples projected onto a new space.
        """

        self.fit(x, y)

        return self.transform(x)
