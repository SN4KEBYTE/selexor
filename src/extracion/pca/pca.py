import numpy as np


class PCA:
    def __init__(self, k: int) -> None:
        self.__k: int = k
        self.__w: np.ndarray = None
        self.__variance_explained: list = None

    def fit(self, x: np.ndarray) -> 'PCA':
        cov_mat = np.cov(x.T)
        eigen_vals, eigen_vec = np.linalg.eig(cov_mat)
        eigen_pairs = sorted([(np.abs(eigen_vals[i]), eigen_vec[:, i]) for i in range(len(eigen_vals))], reverse=True)

        eigen_sum = np.sum(eigen_vals)
        self.__variance_explained = [val / eigen_sum for val in eigen_vals]

        new_features = []

        for i in range(self.__k):
            new_features.append(eigen_pairs[i][1][:, np.newaxis])

        self.__w = np.hstack(new_features)

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.__w is None:
            raise RuntimeError('Projection matrix is not calculated. Please use fit method first, or fit_transform.')

        return x.dot(self.__w)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)

        return x.dot(self.__w)

    @property
    def w(self):
        return self.__w

    @property
    def variance_explained(self):
        return self.__variance_explained
