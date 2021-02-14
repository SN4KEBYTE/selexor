from typing import Callable, Dict, Optional

import numpy as np
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
        self.__importances: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'RFSelector':
        """
        A method that fits the dataset in order to select features.

        :param x: samples.
        :param y: class labels.

        :return: fitted selector.
        """

        self.__forest.fit(x, y)
        self.__importances: np.ndarray = self.__forest.feature_importances_
        self._indices: np.ndarray = np.argsort(self.__importances)[::-1]

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
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

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        A method that fits the dataset and applies transformation to a given samples.

        :param x: samples.
        :param y: class labels.

        :return: samples with the most important features.
        """

        self.fit(x, y)

        return self.transform(x)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """
        Feature importances.

        :return: feature importances or None in case fit (or fit_transform) was not called.
        """

        return self.__importances
