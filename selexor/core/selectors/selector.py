from abc import ABC, abstractmethod
from typing import Callable, Optional

from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.core.base.base import Base
from selexor.core.selectors.types import AccuracyScore
from sklearn.metrics import accuracy_score


class Selector(Base, ABC):
    def __init__(self, n_components: int, scoring: AccuracyScore = accuracy_score) -> None:
        super(Selector, self).__init__(n_components)
        self._scoring: AccuracyScore = scoring
        self._indices: Optional[NDArray[Number]] = None

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self) -> NDArray[Number]:
        pass

    @abstractmethod
    def fit_transform(self) -> NDArray[Number]:
        pass
