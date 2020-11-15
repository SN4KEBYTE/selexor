from collections import OrderedDict
from itertools import combinations
from typing import Any, List, OrderedDict as OrdDict, Optional, Union

import numpy as np
from nptyping import Number, Int
from nptyping.ndarray import NDArray
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from selexor.core.selectors.selector import Selector
from selexor.core.selectors.types import AccuracyScore
from selexor.sbs.types import Subset, FeatureSet, Proportion, RandomState

POSSIBLE_SELECTION_OPTIONS: List[str] = ['score', 'size']


class SBS(Selector):
    """
    Sequential backward selection algorithm.
    """

    def __init__(self, estimator: Any, n_components: int, scoring: AccuracyScore = accuracy_score,
                 test_size: Proportion = 0.3,
                 random_state: RandomState = 0,
                 option: str = 'score', best: bool = True) -> None:
        """
        Initialize the class with some values.

        :param estimator: the estimator for which you want to select features.
        :param n_components: desired number of features.
        :param scoring: accuracy classification score.
        :param test_size: represents the proportion of the dataset to include in the test split.
        :param random_state: controls the shuffling applied to the data before applying the split.
        :param option: selection the proper feature set option (by score or by size).
        :param best: if True, the best feature set will be selected. If False, the worst feature set will be selected.

        :return: None

        :raises: ValueError: thrown when some arguments are not valid.
        """

        super(SBS, self).__init__(n_components, scoring)
        self.__estimator: Any = clone(estimator)

        if type(test_size) == 'float' and 0.0 >= test_size >= 1.0 or type(test_size) == 'int' and test_size < 0:
            raise ValueError('If float, test size must be between 0.0 and 1.0. If int, test size must be positive.')
        self.__test_size: float = test_size

        self.__random_state: Optional[Union[int, np.random.RandomState]] = random_state
        self.__feature_sets: Optional[OrdDict[int, FeatureSet]] = None
        self.__indices: Optional[Subset] = None

        if option not in POSSIBLE_SELECTION_OPTIONS:
            raise ValueError(f"Possible selection options are: {', '.join(POSSIBLE_SELECTION_OPTIONS)}.")
        self.__option = option

        self.__best = best

    def fit(self, x: NDArray[Number], y: NDArray[Number]) -> 'SBS':
        """
        A method that fits the dataset in order to select features.

        :param x: samples.
        :param y: class labels.

        :return: fitted selector.

        :raises: RuntimeError: thrown when test size is bigger than number of samples.
        """

        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.__test_size,
                                                                random_state=self.__random_state)
        except ValueError as val_err:
            raise RuntimeError('Invalid test size.') from val_err

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

        self.__select_feature_set()

        return self

    def transform(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        A method that transforms the samples by selecting the most important features.

        :param x: samples.

        :return: features projected onto a new space.

        :raises: RuntimeError: thrown when feature sets are not calculated. In this case you need to use fit method
                               first (or fit_transform).
        """

        if self.__feature_sets is None:
            raise RuntimeError('Feature sets are not calculated. Please use fit method first, or fit_transform.')

        return x[:, self.__indices]

    def fit_transform(self, x: NDArray[Number], y: NDArray[Number]) -> NDArray[Number]:
        """
        A method that fits the dataset and applies transformation to a given samples.

        :param x: samples.
        :param y: class labels.

        :return: samples projected onto a new space.
        """

        self.fit(x, y)

        return self.transform(x)

    def __select_feature_set(self) -> None:
        """
        A method that selects the proper feature set from the most perspective feature sets.

        :return: None.
        """

        # if we want to choose the best feature set based on the size
        if self.__option == 'size':
            # if self.__best is True, we will get the biggest feature set, smallest otherwise
            funcs = {True: max, False: min}

            # find the proper size, get the appropriate feature set (we use [0] at the end because we also store
            # accuracy score)
            self.__indices = self.__feature_sets[funcs[self.__best](self.__feature_sets.keys())][0]
        else:
            # get feature sets sorted by accuracy score in ascending order
            sets = sorted(self.__feature_sets.values(), key=lambda t: t[1])

            # if self.__best is True, we need the last feature set (with the biggest classification score), first
            # otherwise
            self.__indices = sets[-1 if self.__best else 0][0]

    def __calculate_score(self, x_train: NDArray[Number], y_train: NDArray[Number], x_test: NDArray[Number],
                          y_test: NDArray[Number], indices: Subset) -> float:
        """
        A method that calculates classification score on provided features.

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
