from typing import List, Tuple, Optional, Union

import numpy as np

Subset = Tuple[int, ...]
FeatureSet = Tuple[List[Subset], float]
RandomState = Optional[Union[int, np.random.RandomState]]
Proportion = Optional[Union[float, int]]
