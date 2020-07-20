from collections import OrderedDict
from itertools import combinations
from typing import Tuple, List, Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# todo: find a right way to type hint estimator
class SBS:
    def __init__(self, estimator, k_features: int, scoring: Callable = accuracy_score,
                 test_size: float = 0.3, random_state: int = 1) -> None:
        self.__estimator = clone(estimator)
        self.__k_features: int = k_features
        self.__scoring: Callable = scoring
        self.__test_size: float = test_size
        self.__random_state: int = random_state
        self.__indices: Tuple[int, ...] = ()
        self.__subsets: List[Tuple[int, ...]] = []
        self.__scores: List[int, ...] = []

    # todo: find a way to type hint return value type without conflicts
    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.__test_size,
                                                            random_state=self.__random_state)

        dim: int = x_train.shape[1]
        self.__indices = tuple(range(dim))
        self.__subsets = [self.__indices]

        score: float = self.__calculate_score(x_train, y_train, x_test, y_test, self.__indices)
        self.__scores = [score]

        while dim > self.__k_features:
            scores: List[float, ...] = []
            subsets: List[Tuple[int, ...]] = []

            for p in combinations(self.__indices, dim - 1):
                score = self.__calculate_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best: np.int64 = np.argmax(scores)
            self.__indices = subsets[best]
            self.__subsets.append(self.__indices)
            dim -= 1

            self.__scores.append(scores[best])

        return OrderedDict(sorted({len(s): (s, self.__scores[i]) for i, s in enumerate(self.__subsets)}.items(),
                                  key=lambda t: t[0]))

    @staticmethod
    def __transform(x: np.ndarray, indices: Tuple[int, ...]) -> np.ndarray:
        return x[:, indices]

    def __calculate_score(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                          indices: Tuple[int, ...]) -> float:
        self.__estimator.fit(self.__transform(x_train, indices), y_train)
        y_pred = self.__estimator.predict(self.__transform(x_test, indices))

        return self.__scoring(y_test, y_pred)
