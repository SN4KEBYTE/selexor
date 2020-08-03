from typing import Callable, Dict

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from selexor.core.selectors.selector import Selector


class RFSelector(Selector):
    def __init__(self, n_components: int, estimator_params: Dict,
                 scoring: Callable = accuracy_score) -> None:
        super(RFSelector, self).__init__(n_components, scoring)
        self.__forest: RandomForestClassifier = RandomForestClassifier(**estimator_params)
        self.__importances: NDArray[Number] or None = None

    def fit(self, x_train: NDArray[Number], y_train: NDArray[Number]) -> 'RFSelector':
        self.__forest.fit(x_train, y_train)
        self.__importances: NDArray[Number] = self.__forest.feature_importances_
        self._indices: NDArray[Number] = np.argsort(self.__importances)[::-1]

        return self

    def transform(self, x: NDArray[Number]) -> NDArray[Number]:
        if self.__importances is None:
            raise RuntimeError('Feature importances are not calculated. Please use fit method first, or fit_transform.')

        return self._indices[:self._n_components]

    def fit_transform(self, x_train: NDArray[Number], y_train: NDArray[Number]) -> NDArray[Number]:
        self.fit(x_train, y_train)

        return self.transform(x_train)

    @property
    def features_importances(self) -> NDArray[Number]:
        return self.__importances
