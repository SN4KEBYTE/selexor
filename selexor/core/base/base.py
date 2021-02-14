from abc import ABC, abstractmethod

import numpy as np


class Base(ABC):
    """
    Abstract base class for all selectors and extractors.
    """

    def __init__(self, n_components: int) -> None:
        """
        Initialize the class with some values.

        :param n_components: desired number of components.

        :return: None.
        """

        self._n_components: int = n_components

    @abstractmethod
    def fit(self, *args, **kwargs) -> 'Base':
        """
        A method that fits the dataset in order to select (or extract) features. This is an abstract method and must be
        implemented in subclasses.

        :param args: variable length argument list.
        :param kwargs: arbitrary keyword arguments.

        :return: fitted algorithm.
        """

        pass

    @abstractmethod
    def transform(self, *args, **kwargs) -> np.ndarray:
        """
        A method that transforms the samples. This is an abstract method and must be implemented in subclasses.

        :param args: variable length argument list.
        :param kwargs: arbitrary keyword arguments.

        :return: transformed samples.
        """

        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """
        A method that fits the dataset and applies transformation to a given samples. This is an abstract method and
        must be implemented in subclasses.

        :param args: variable length argument list.
        :param kwargs: arbitrary keyword arguments.

        :return: transformed samples.
        """

        pass
