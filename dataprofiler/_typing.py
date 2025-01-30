"""Contains typing aliases."""

from typing import NewType, Union

import numpy as np
import pandas as pd

DataArray = Union[pd.DataFrame, pd.Series, np.ndarray]
JSONType = Union[str, int, float, bool, None, list, dict]
Url = NewType("Url", str)
