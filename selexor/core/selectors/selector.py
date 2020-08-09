from abc import ABC, abstractmethod
from typing import Optional

from nptyping import Number
from nptyping.ndarray import NDArray
from sklearn.metrics import accuracy_score

from selexor.core.base.base import Base
from selexor.core.selectors.types import AccuracyScore


class Selector(Base, ABC):
    def __init__(self, n_components: int, scoring: AccuracyScore = accuracy_score) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired number of features.
        :param scoring: accuracy classification score.

        :return: None.
        """

        super(Selector, self).__init__(n_components)
        self._scoring: AccuracyScore = scoring
        self._indices: Optional[NDArray[Number]] = None

    @abstractmethod
    def fit(self, x: NDArray[Number], y: NDArray[Number]) -> 'Selector':
        """
        A method that fits the dataset in order to select features. This is an abstract method and must be implemented
        in subclasses.

        :param x: samples.
        :param y: class labels.

        :return: fitted selector.
        """

        pass

    @abstractmethod
    def transform(self, x: NDArray[Number]) -> NDArray[Number]:
        """
        A method that transforms the samples by selecting the most important features. This is an abstract method and
        must be implemented in subclasses.

        :param x: samples.

        :return: samples with the most important features.
        """

        pass

    @abstractmethod
    def fit_transform(self, x: NDArray[Number], y: NDArray[Number]) -> NDArray[Number]:
        """
        A method that fits the dataset and applies transformation to a given samples. This is an abstract method and
        must be implemented in subclasses.

        :param x: samples.
        :param y: class labels.

        :return: samples with the most important features.
        """

        pass
