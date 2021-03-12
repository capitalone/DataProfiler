
class AbstractTestOptions():
	
	@classmethod
	def get_options(self, *args, **kwargs):
		if self.option_class == None:
			raise ValueError("option_class class variable cannot be set to 'None'")
		return self.option_class(*args, **kwargs)

	@classmethod
	def get_options_path(self, *args, **kwargs): 
		if self.option_class == None:	
			raise ValueError("option_class class variable cannot be set to 'None'")
		return self.option_class.__name__
