"""Contains methods to decode components of a Profiler."""

from . import categorical_column_profile


def decode_categorical_column(to_decode: dict):
    """
    Specify how CategoricalColumn should be deserialized.

    :param to_decode: an object to be deserialized
    :type to_serialize: a dictionary resullting from json.loads()
    :return: CategoricalColumn object
    """
    decoded = categorical_column_profile.CategoricalColumn(to_decode["name"])
    for attr, value in to_decode.items():
        decoded.__setattr__(attr, value)
