from typing import Dict

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from selexor.core.selectors.selector import Selector


class RFSelector(Selector):
    def __init__(self, n_components: int, estimator_params: Dict,
                 scoring: callable = accuracy_score) -> None:
        super(RFSelector, self).__init__(n_components, scoring)
        self.__forest = RandomForestClassifier(**estimator_params)
        self.__importances = None

    def fit(self, x_train: NDArray[Number], y_train: NDArray[Number]) -> 'RFSelector':
        self.__forest.fit(x_train, y_train)
        self.__importances = self.__forest.feature_importances_
        self._indices = np.argsort(self.__importances)[::-1]

        return self

    def transform(self) -> NDArray[Number]:
        pass

    def fit_transform(self) -> NDArray[Number]:
        pass

    def select(self) -> np.ndarray:
        if self.__importances is None:
            self.__forest.fit(self.__x_train, self.__y_train)
            self.__importances = self.__forest.feature_importances_
            self._indices = np.argsort(self.__importances)[::-1]

        return self.__indices[:self.__k_features]

    @property
    def features_importances(self):
        return self.__importances
