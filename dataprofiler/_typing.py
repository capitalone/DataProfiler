"""Contains typing aliases."""
from typing import Dict, List, NewType, Union

import numpy as np
import pandas as pd

DataArray = Union[pd.DataFrame, pd.Series, np.ndarray]
JSONType = Union[str, int, float, bool, None, List, Dict]
Url = NewType("Url", str)
