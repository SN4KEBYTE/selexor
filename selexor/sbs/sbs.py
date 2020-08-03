from collections import OrderedDict
from itertools import combinations
from typing import List, Callable, OrderedDict as OrdDict

import numpy as np
from nptyping import Number, Int
from nptyping.ndarray import NDArray
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from selexor.core.selectors.selector import Selector
from selexor.sbs.types import Subset, FeatureSet


# todo: find a right way to type hint estimator
class SBS(Selector):
    def __init__(self, estimator, n_components: int, scoring: Callable = accuracy_score,
                 test_size: float = 0.3, random_state: int = 0) -> None:
        """
        Initialize the class with some values.

        :param estimator: the estimator for which you want to select features.
        :param n_components: desired number of features.
        :param scoring: accuracy classification score.
        :param test_size: represents the proportion of the dataset to include in the test split.
        :param random_state: controls the shuffling applied to the data before applying the split.
        :return: None
        """

        super(SBS, self).__init__(n_components, scoring)

        self.__estimator = clone(estimator)
        self.__test_size: float = test_size
        self.__random_state: int = random_state
        self.__indices: Subset or None = None
        self.__subsets: List[Subset] or None = None
        self.__scores: List[int, ...] or None = None
        self.__feature_sets: OrdDict[int, FeatureSet] or None = None

    def fit(self, x: NDArray[Number], y: NDArray[Number]) -> 'SBS':
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

        while dim > self._n_components:
            scores: List[float, ...] = []
            subsets: List[Subset] = []

            for p in combinations(self.__indices, dim - 1):
                score = self.__calculate_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best: NDArray[Int] = np.argmax(scores)
            self.__indices = subsets[best]
            self.__subsets.append(self.__indices)
            dim -= 1

            self.__scores.append(scores[best])

        self.__feature_sets = OrderedDict(
            sorted({len(s): (s, self.__scores[i]) for i, s in enumerate(self.__subsets)}.items(),
                   key=lambda t: t[0]))

        return self

    def transform(self) -> NDArray[Number]:
        pass

    def fit_transform(self) -> NDArray[Number]:
        pass

    # todo: this function must select the most perspective feature set (based on size or score).
    # you can't use transform without selecting definite feature set.
    def select_feature_set(self, option: str = 'score'):
        pass

    @staticmethod
    def __transform(x: NDArray[Number], indices: Subset) -> NDArray[Number]:
        """
        An auxiliary function which takes definite columns from NumPy array.

        :param x: array to be transformed.
        :param indices: indices of required columns.

        :return: transformed array.
        """

        return x[:, indices]

    def __calculate_score(self, x_train: NDArray[Number], y_train: NDArray[Number], x_test: NDArray[Number],
                          y_test: NDArray[Number],
                          indices: Subset) -> float:
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

        return self._scoring(y_test, y_pred)
