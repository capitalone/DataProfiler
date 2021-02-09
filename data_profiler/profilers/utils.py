# Recursive dictionary merge
# Copyright (C) 2016 Paul Durivage <pauldurivage+github@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import collections
import random
import math


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


class KeyDict(collections.defaultdict):
    """
    Helper class for sample_in_chunks. Allows keys that are missing to become
    the values for that key.
    From:
    https://www.drmaciver.com/2018/01/lazy-fisher-yates-shuffling-for-precise-rejection-sampling/
    """
    def __missing__(self, key):
        return key


def shuffle_in_chunks(data_length, chunk_size):
    """
    A generator for creating shuffled indexes in chunks. This reduces the cost
    of having to create all indexes, but only of that what is needed.
    Initial Code idea from:
    https://www.drmaciver.com/2018/01/lazy-fisher-yates-shuffling-for-precise-rejection-sampling/
    
    :param data_length: length of data to be shuffled
    :param chunk_size: size of shuffled chunks
    :return: list of shuffled indices of chunk size
    """
    indices = KeyDict()
    j = 0
    # loop through all chunks
    for chunk_ind in range(max(math.ceil(data_length / chunk_size), 1)):

        # determine the chunk size and preallocate an array
        true_chunk_size = min(chunk_size, data_length - chunk_size * chunk_ind)
        values = [-1] * true_chunk_size

        # shuffle the indexes
        for count in range(true_chunk_size):

            # get a random index to swap and swap it with j
            k = random.randrange(j, data_length)
            indices[j], indices[k] = indices[k], indices[j]

            # set the swapped value to the output
            values[count] = indices[j]

            # increment so as not to include values already swapped
            j += 1
            
        yield values
