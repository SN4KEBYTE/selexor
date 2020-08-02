from abc import ABC, abstractmethod
from typing import Dict, Callable

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

from selexor.extraction.base_extractor import BaseExtractor


class KernelExtractor(BaseExtractor, ABC):
    def __init__(self, n_components: int, gamma: float = 1.0, kernel: str = 'rbf') -> None:
        super(KernelExtractor, self).__init__(n_components)
        self._gamma = gamma
        self._kernel_func: Callable = self.__get_kernel(kernel)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self) -> NDArray[Number]:
        pass

    @abstractmethod
    def fit_transform(self) -> NDArray[Number]:
        pass

    def __get_kernel(self, kernel: str) -> Callable:
        kernels: Dict = {'rbf': self.__rbf}

        return kernels[kernel]

    def __rbf(self, mat_dists: NDArray[Number]) -> NDArray[Number]:
        return np.exp(-self._gamma * mat_dists)
