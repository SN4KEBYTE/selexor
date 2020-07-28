import numpy as np

from src.extracion.extractor import Extractor


class PCA(Extractor):
    def __init__(self, k: int) -> None:
        super(PCA, self).__init__(k)

    def fit(self, x_train: np.ndarray) -> 'PCA':
        cov_mat = np.cov(x_train.T)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

        self._calculate_variance_explained(eigen_vals)
        self._calculate_projection_matrix(eigen_vals, eigen_vecs)

        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)

        return self.transform(x)
