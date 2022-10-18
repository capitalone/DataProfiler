"""Contains typing aliases."""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

DataArray = Union[pd.DataFrame, pd.Series, np.ndarray]
JSONType = Union[str, int, float, bool, None, List["JSONType"], Dict[str, "JSONType"]]
