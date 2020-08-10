from typing import Callable, List, Tuple, Optional, Union

import numpy as np
from nptyping import Number
from nptyping.ndarray import NDArray

Subset = Tuple[int, ...]
FeatureSet = Tuple[List[Subset], float]
AccuracyScore = Callable[[NDArray[Number], NDArray[Number]], float]
RandomState = Optional[Union[int, np.random.RandomState]]
Proportion = Optional[Union[float, int]]
