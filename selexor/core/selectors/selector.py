from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score

from selexor.core.base.base import Base
from selexor.core.selectors.types import AccuracyScore


class Selector(Base, ABC):
    """
    Abstract base class for all selectors.
    """

    def __init__(self, n_components: int, scoring: AccuracyScore = accuracy_score) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired number of features.
        :param scoring: accuracy classification score.

        :return: None.
        """

        super(Selector, self).__init__(n_components)
        self._scoring: AccuracyScore = scoring
        self._indices: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'Selector':
        """
        A method that fits the dataset in order to select features. This is an abstract method and must be implemented
        in subclasses.

        :param x: samples.
        :param y: class labels.

        :return: fitted selector.
        """

        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        A method that transforms the samples by selecting the most important features. This is an abstract method and
        must be implemented in subclasses.

        :param x: samples.

        :return: samples with the most important features.
        """

        pass

    @abstractmethod
    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        A method that fits the dataset and applies transformation to a given samples. This is an abstract method and
        must be implemented in subclasses.

        :param x: samples.
        :param y: class labels.

        :return: samples with the most important features.
        """

        pass
