from typing import Callable

import numpy as np

AccuracyScore = Callable[[np.ndarray, np.ndarray], float]
