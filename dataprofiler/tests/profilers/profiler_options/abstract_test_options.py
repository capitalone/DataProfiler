import json

from dataprofiler.profilers.json_decoder import load_option
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import BaseOption

from .. import utils as test_utils


class AbstractTestOptions:

    option_class = None

    @classmethod
    def validate_option_class(cls, *args, **kwargs):
        # Check option_class was set
        if cls.option_class is None:
            raise ValueError("option_class class variable cannot be set to " "'None'")

        # Check option_class is correct type
        if not isinstance(cls.option_class, type):
            raise ValueError("option_class class variable must be of type " "'type'")

        # Check option_class() is correct type
        options = cls.option_class(*args, **kwargs)
        if not isinstance(options, BaseOption):
            raise ValueError(
                "option_class class variable must create object " "of type 'BaseOption'"
            )

    @classmethod
    def get_options(cls, *args, **kwargs):
        cls.validate_option_class(*args, **kwargs)
        return cls.option_class(*args, **kwargs)

    @classmethod
    def get_options_path(cls, *args, **kwargs):
        cls.validate_option_class(*args, **kwargs)
        return cls.option_class.__name__

    def test_init(self):
        raise NotImplementedError

    def test_set_helper(self):
        raise NotImplementedError

    def test_set(self):
        raise NotImplementedError

    def test_validate_helper(self):
        raise NotImplementedError

    def test_validate(self):
        raise NotImplementedError


class JSONDecodeTestMixin:
    def test_json_decode(self):
        expected_options = self.get_options()

        serialized = json.dumps(expected_options, cls=ProfileEncoder)
        deserialized = load_option(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_options)
