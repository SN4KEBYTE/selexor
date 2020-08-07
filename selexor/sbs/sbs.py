from collections import OrderedDict
from itertools import combinations
from typing import Any, List, OrderedDict as OrdDict, Optional

import numpy as np
from nptyping import Number, Int
from nptyping.ndarray import NDArray
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from selexor.core.selectors.selector import Selector
from selexor.core.selectors.types import AccuracyScore
from selexor.sbs.types import Subset, FeatureSet


class SBS(Selector):
    def __init__(self, estimator: Any, n_components: int, scoring: AccuracyScore = accuracy_score,
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
        self.__estimator: Any = clone(estimator)
        self.__test_size: float = test_size
        self.__random_state: int = random_state
        self.__feature_sets: Optional[OrdDict[int, FeatureSet]] = None
        self.__indices: Optional[Subset] = None

    def fit(self, x: NDArray[Number], y: NDArray[Number], option: str = 'score') -> 'SBS':
        """
        A method that fits the dataset in order to select features.

        :param option: todo
        :param x: samples.
        :param y: class labels.

        :return: the most perspective feature sets.
        """

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.__test_size,
                                                            random_state=self.__random_state)

        dim: int = x_train.shape[1]
        indices: Subset = tuple(range(dim))
        best_subsets: List[Subset] = [indices]

        score: float = self.__calculate_score(x_train, y_train, x_test, y_test, indices)
        best_scores: List[float] = [score]

        while dim > self._n_components:
            scores: List[float] = []
            subsets: List[Subset] = []

            for p in combinations(indices, dim - 1):
                score = self.__calculate_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best: NDArray[Int] = np.argmax(scores)
            indices = subsets[best]
            best_subsets.append(indices)
            dim -= 1

            best_scores.append(scores[best])

        self.__feature_sets = OrderedDict(
            sorted({len(s): (s, best_scores[i]) for i, s in enumerate(best_subsets)}.items(),
                   key=lambda t: t[0]))

        self.select_feature_set(option)

        return self

    def transform(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        A method that applies dimensionality reduction to a given array.

        :param x: samples.
        :return: features projected onto a new space.

        :raises: RuntimeError: thrown when feature sets are not calculated. In this case you need to use fit method
                               first (or fit_transform).
        """

        if self.__feature_sets is None:
            raise RuntimeError('Feature sets are not calculated. Please use fit method first, or fit_transform.')

        return x[:, self.__indices]

    def fit_transform(self, x: NDArray[Number], y: NDArray[Number], option: str = 'score') -> NDArray[Number]:
        """
        A method that fits the dataset and applies dimensionality reduction to a given array.

        :param x: samples.
        :param y: class labels.
        :param option: todo
        :return: features projected onto a new space.
        """

        self.fit(x, y, option)

        return self.transform(x)

    def select_feature_set(self, option: str = 'score', best: bool = True) -> 'SBS':
        if option == 'size':
            funcs = {True: max, False: min}

            self.__indices = self.__feature_sets[funcs[best](self.__feature_sets.keys())][1]
        else:
            # todo: find a way to get subset with the highest score
            pass
            # keys = self.__feature_sets.keys()
            # items = sorted(self.__feature_sets.items(), key=lambda t: t[1])
            #
            # tmp = OrderedDict({subset[1]: tuple(subset[0], score) for score, subset in keys, items})
            # 
            # print(keys)
            # print(items)
            #
            # # self.__indices = feature_sets_copy[0 if best else -1]

        return self

    def __calculate_score(self, x_train: NDArray[Number], y_train: NDArray[Number], x_test: NDArray[Number],
                          y_test: NDArray[Number], indices: Subset) -> float:
        """
        A function that calculates classification score on provided features.

        :param x_train: train samples.
        :param y_train: train class labels.
        :param x_test: test samples.
        :param y_test: test class labels.
        :param indices: indices of features for which you want to calculate score.

        :return: classification score.
        """

        self.__estimator.fit(x_train[:, indices], y_train)
        y_pred = self.__estimator.predict(x_test[:, indices])

        return self._scoring(y_test, y_pred)

    @property
    def feature_sets(self) -> Optional[OrdDict[int, FeatureSet]]:
        """
        The most perspective feature sets.

        :return: the most perspective feature sets or None in case fit (or fit_transform) was not called.
        """
        
        return self.__feature_sets
