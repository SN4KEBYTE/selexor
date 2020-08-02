from typing import List

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from src.extraction.linear_extractor import LinearExtractor


class LDA(LinearExtractor):
    def __init__(self, k: int) -> None:
        super(LDA, self).__init__(k)

    def fit(self, x_train: NDArray[Number], y_train: NDArray[Number]) -> 'LDA':
        labels: NDArray[Number] = np.sort(np.unique(y_train))

        mean_vecs: List[NDArray[Number]] = [np.mean(x_train[y_train == label], axis=0) for label in labels]

        dim: int = x_train.shape[1]
        s_w: NDArray[Number] = np.zeros((dim, dim))

        for label, mean_vec in zip(labels, mean_vecs):
            class_scatter: NDArray[Number] = np.cov(x_train[y_train == label].T)
            s_w += class_scatter

        mean_overall: NDArray[Number] = np.mean(x_train, axis=0)
        s_b: NDArray[Number] = np.zeros((dim, dim))

        for i, mean_vec in enumerate(mean_vecs):
            n: int = x_train[y_train == i, :].shape[0]
            mean_vec: NDArray[Number] = mean_vec.reshape(dim, 1)
            mean_overall: NDArray[Number] = mean_overall.reshape(dim, 1)
            s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))

        self._calculate_variance_explained(eigen_vals)
        self._calculate_projection_matrix(eigen_vals, eigen_vecs)

        return self

    def fit_transform(self, x: NDArray[Number], y_train: NDArray[Number]) -> NDArray[Number]:
        self.fit(x, y_train)

        return self.transform(x)
