from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


class RFSelector:
    def __init__(self, estimator_params: dict, k_features: int, x_train, y_train, scoring: callable = accuracy_score,
                 test_size: float = 0.2, random_state: int = 1) -> None:
        self.__k_features = k_features
        self.__x_train = x_train
        self.__y_train = y_train
        self.__scoring = scoring
        self.__test_size = test_size
        self.__random_state = random_state
        self.__forest = RandomForestClassifier(**estimator_params)

    def plot(self):
        pass

    def select(self):
        self.__forest.fit(self.__x_train, self.__y_train)
        importances = self.__forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        return indices[:self.__k_features]
