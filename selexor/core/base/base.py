from abc import ABC, abstractmethod

from nptyping import Number
from nptyping.ndarray import NDArray


class Base(ABC):
    def __init__(self, n_components: int) -> None:
        self._n_components: int = n_components

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self) -> NDArray[Number]:
        pass

    @abstractmethod
    def fit_transform(self) -> NDArray[Number]:
        pass
