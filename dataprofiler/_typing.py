"""Contains typing aliases."""
from typing import Union

import numpy as np
import pandas as pd

DataArray = Union[pd.DataFrame, pd.Series, np.ndarray]
