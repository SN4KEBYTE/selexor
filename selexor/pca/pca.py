import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.core.extractors.linear_extractor import LinearExtractor


class PCA(LinearExtractor):
    """
    Principal component analysis extractor.
    """

    def __init__(self, n_components: int) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired dimension of the new feature space.

        :return: None
        """

        super(PCA, self).__init__(n_components)

    def fit(self, x: NDArray[Number]) -> 'PCA':
        """
        A method that fits the dataset in order to extract features.

        :param x: samples.

        :return: fitted extractor.
        """

        cov_mat: NDArray[Number] = np.cov(x.T)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

        self._calculate_explained_variance(eigen_vals)
        self._calculate_projection_matrix(eigen_vals, eigen_vecs)

        return self

    def fit_transform(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        A method that fits the dataset and applies dimensionality reduction to a given samples.

        :param x: samples.

        :return: samples projected onto a new space.
        """

        self.fit(x)

        return self.transform(x)
