import os
import shutil
import random

import numpy as np

import dataprofiler as dp


def set_seed(seed=None):
    """
    Sets the see for all possible random state libraries
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    dp.set_seed(seed)