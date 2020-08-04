from typing import Callable

from nptyping import Number
from nptyping.ndarray import NDArray

AccuracyScore = Callable[[NDArray[Number], NDArray[Number]], float]
