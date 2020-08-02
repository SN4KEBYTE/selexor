from collections import OrderedDict
from itertools import combinations
from typing import Tuple, List, Callable, OrderedDict as OrdDict

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# todo: find a right way to type hint estimator
class SBS:
    def __init__(self, estimator, k_features: int, *args, scoring: Callable = accuracy_score,
                 test_size: float = 0.3, random_state: int = 1, **kwargs) -> None:
        """
        Initialize the class with some values.

        :param estimator: the estimator for which you want to select features.
        :param k_features: desired number of features.
        :param args: variable length argument list.
        :param scoring: accuracy classification score.
        :param test_size: represents the proportion of the dataset to include in the test split.
        :param random_state: controls the shuffling applied to the data before applying the split.
        :param kwargs: arbitary keyword arguments.

        :return: None
        """

        self.__estimator = clone(estimator)
        self.__k_features: int = k_features
        self.__scoring: Callable = scoring
        self.__test_size: float = test_size
        self.__random_state: int = random_state
        self.__indices: Tuple[int, ...] = ()
        self.__subsets: List[Tuple[int, ...]] = []
        self.__scores: List[int, ...] = []

    # todo: find a way to type hint return value type without conflicts
    def fit(self, x: NDArray[Number], y: NDArray[Number]) -> OrdDict[int, Tuple[List[Tuple[int, ...]], float]]:
        """
        A method that fits the dataset in order to select features.

        :param x: features.
        :param y: class labels.

        :return: OrderedDict with the most perspective feature sets.
        """

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
    def __transform(x: NDArray[Number], indices: Tuple[int, ...]) -> NDArray[Number]:
        """
        An auxiliary function which takes definite columns from NumPy array.

        :param x: array to be transformed.
        :param indices: indices of required columns.

        :return: transformed array.
        """

        return x[:, indices]

    def __calculate_score(self, x_train: NDArray[Number], y_train: NDArray[Number], x_test: NDArray[Number], y_test: NDArray[Number],
                          indices: Tuple[int, ...]) -> float:
        """
        A function that calculates classification score on provided features.

        :param x_train: train samples.
        :param y_train: train class labels.
        :param x_test: test samples.
        :param y_test: test class labels.
        :param indices: indices of features for which you want to calculate score.

        :return: classification score.
        """

        self.__estimator.fit(self.__transform(x_train, indices), y_train)
        y_pred = self.__estimator.predict(self.__transform(x_test, indices))

        return self.__scoring(y_test, y_pred)
