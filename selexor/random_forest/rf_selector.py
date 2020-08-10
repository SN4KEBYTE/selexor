from typing import Callable, Dict, Optional

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from selexor.core.selectors.selector import Selector


class RFSelector(Selector):
    """
    Random forest selector.
    """

    def __init__(self, n_components: int, estimator_params: Dict,
                 scoring: Callable = accuracy_score) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired number of features.
        :param estimator_params: params for RandomForestClassifier.
               Visit https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for
               more info.
        :param scoring: accuracy classification score.

        :return: None.
        """

        super(RFSelector, self).__init__(n_components, scoring)
        self.__forest: RandomForestClassifier = RandomForestClassifier(**estimator_params)
        self.__importances: Optional[NDArray[Number]] = None

    def fit(self, x: NDArray[Number], y: NDArray[Number]) -> 'RFSelector':
        """
        A method that fits the dataset in order to select features.

        :param x: samples.
        :param y: class labels.

        :return: fitted selector.
        """

        self.__forest.fit(x, y)
        self.__importances: NDArray[Number] = self.__forest.feature_importances_
        self._indices: NDArray[Number] = np.argsort(self.__importances)[::-1]

        return self

    def transform(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        A method that transforms the samples by selecting the most important features.

        :param x: samples.

        :return: samples with the most important features.

        :raises: RuntimeError: thrown when the most important features are not calculated. In this case you need to use
                 fit method first (or fit_transform).
        """

        if self.__importances is None:
            raise RuntimeError('Feature importances are not calculated. Please use fit method first, or fit_transform.')

        return x[:, self._indices[:self._n_components]]

    def fit_transform(self, x: NDArray[Number], y: NDArray[Number]) -> NDArray[Number]:
        """
        A method that fits the dataset and applies transformation to a given samples.

        :param x: samples.
        :param y: class labels.

        :return: samples with the most important features.
        """

        self.fit(x, y)

        return self.transform(x)

    @property
    def feature_importances(self) -> Optional[NDArray[Number]]:
        """
        Feature importances.

        :return: feature importances or None in case fit (or fit_transform) was not called.
        """

        return self.__importances
