"""Create a random number generator using a manual seed DATAPROFILER_SEED."""
import os
import warnings

import numpy as np

from . import settings


def get_random_number_generator() -> np.random._generator.Generator:
    """Create a random number generator using a manual seed DATAPROFILER_SEED."""
    rng = np.random.default_rng(settings._seed)
    if "DATAPROFILER_SEED" in os.environ and settings._seed is None:
        seed: str = os.environ.get("DATAPROFILER_SEED", "")
        try:
            rng = np.random.default_rng(int(seed))
        except ValueError:
            warnings.warn("Seed should be an integer", RuntimeWarning)
    return rng
