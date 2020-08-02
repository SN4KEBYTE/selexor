import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from src.extraction.linear_extractor import LinearExtractor


class PCA(LinearExtractor):
    def __init__(self, k: int) -> None:
        super(PCA, self).__init__(k)

    def fit(self, x_train: NDArray[Number]) -> 'PCA':
        cov_mat: NDArray[Number] = np.cov(x_train.T)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

        self._calculate_variance_explained(eigen_vals)
        self._calculate_projection_matrix(eigen_vals, eigen_vecs)

        return self

    def fit_transform(self, x: NDArray[Number]) -> NDArray[Number]:
        self.fit(x)

        return self.transform(x)
