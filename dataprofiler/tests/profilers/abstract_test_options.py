from dataprofiler.profilers.profiler_options import BaseOption

class AbstractTestOptions():
    
    option_class = None

    @classmethod
    def validate_option_class(self, *args, **kwargs):
        # Check option_class was set
        if self.option_class == None:
            raise ValueError("option_class class variable cannot be set to 'None'")

        # Check option_class is correct type
        if not isinstance(self.option_class, type):
            raise ValueError("option_class class variable must be of type 'type'")

        # Check option_class() is correct type
        options = self.option_class(*args, **kwargs)
        if not isinstance(options, BaseOption):
            raise ValueError("option_class class variable must create object of type 'BaseOption'")
        
    @classmethod
    def get_options(self, *args, **kwargs):
        self.validate_option_class(*args, **kwargs)
        return self.option_class(*args, **kwargs)

    @classmethod
    def get_options_path(self, *args, **kwargs): 
        self.validate_option_class(*args, **kwargs)
        return self.option_class.__name__

    def test_init(self, *mocks):
        raise NotImplementedError
 
    def test_set_helper(self, *mocks):
        raise NotImplementedError
 
    def test_set(self, *mocks):
        raise NotImplementedError
 
    def test_validate_helper(self, *mocks):
        raise NotImplementedError
 
    def test_validate(self, *mocks):
        raise NotImplementedError
