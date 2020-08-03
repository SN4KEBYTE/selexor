from abc import ABC, abstractmethod
from typing import Callable

from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.core.base.base import Base
from sklearn.metrics import accuracy_score


class Selector(Base, ABC):
    def __init__(self, n_components: int, scoring: Callable = accuracy_score) -> None:
        super(Selector, self).__init__(n_components)
        self._scoring: Callable = scoring
        self._indices: NDArray[Number] or None = None

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self) -> NDArray[Number]:
        pass

    @abstractmethod
    def fit_transform(self) -> NDArray[Number]:
        pass
