from typing import Callable, List, Tuple

from nptyping import Number
from nptyping.ndarray import NDArray

Subset = Tuple[int, ...]
FeatureSet = Tuple[List[Subset], float]
AccuracyScore = Callable[[NDArray[Number], NDArray[Number]], float]
