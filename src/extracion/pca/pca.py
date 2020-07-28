import numpy as np

from src.extracion.extractor import Extractor


class PCA(Extractor):
    def __init__(self, k: int) -> None:
        super(PCA, self).__init__(k)

    def fit(self, x: np.ndarray) -> 'PCA':
        cov_mat = np.cov(x.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen_pairs = sorted([(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))],
                             key=lambda k: k[0], reverse=True)

        eigen_sum = np.sum(eigen_vals)
        self._variance_explained = [val / eigen_sum for val in eigen_vals]

        new_features = [eigen_pairs[i][1][:, np.newaxis].real for i in range(self._k)]
        self._w = np.hstack(new_features)

        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)

        return x.dot(self._w)

    @property
    def variance_explained(self):
        return self._variance_explained
