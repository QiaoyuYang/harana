from ..utils import core

# Number of modes used in the model
num_mode = 2
mode_candidates = ["maj", "min"]

# Number of different pitch spellings for key tonics
num_tonic = core.num_pc

num_key = num_tonic * num_mode

class Key:

	def __init__(self, *args, **kwargs):

		self.key_index = None
		self.key_symbol = None
		arg_keys = list(kwargs.keys()) 
		if arg_keys == ["key_symbol"]:
			self.key_symbol = kwargs["key_symbol"]
			self.tonic_ps, self.mode = parse_key_symbol(self.key_symbol)
			self.tonic_pc = core.ps2pc(self.tonic_ps)
			self.mode_index = mode_candidates.index(self.mode)
		elif arg_keys == ["key_index"]:
			self.key_index = kwargs["key_index"]
			self.tonic_pc, self.mode_index = parse_key_index(self.key_index)
			self.tonic_ps = core.pc2ps(self.tonic_index)
			self.mode = mode_candidates[self.mode_index]
		elif arg_keys == ["tonic_ps", "mode"]:
			self.tonic_ps = kwargs["tonic_ps"]
			self.tonic_pc = core.ps2pc(self.tonic_ps)
			self.mode = kwargs["mode"]
			self.mode_index = mode_candidates.index(self.mode)
		elif arg_keys == ["tonic_pc", "mode"]:
			self.tonic_pc = kwargs["tonic_pc"]
			self.tonic_ps = core.pc2ps(self.tonic_ps)
			self.mode = kwargs["mode"]
			self.mode_index = mode_candidates.index(self.mode)

		if not self.key_symbol:
			self.key_symbol = self.get_symbol()
		if not self.key_index:
			self.key_index = self.get_index()


	def __repr__(self):
		return f"Key(root = {self.tonic_ps}, mode = {self.mode})"
	
	def __str__(self):
		return self.get_symbol()
	
	def get_symbol(self):
		return f"{self.tonic_ps}_{self.mode}"

	def get_index(self):
		return num_mode * self.tonic_pc + self.mode_index
	
def parse_key_symbol(key_symbol):
	tonic_ps, mode = key_symbol.split("_")
	return tonic_ps, mode

def parse_key_index(key_index):
	mode_index = key_index % num_mode
	tonic_pc = int((index - mode_index) / num_mode)
	return tonic_pc, mode_index

def key_symbol2index(key_symbol):
	tonic_ps, mode = parse_symbol(symbol)
	tonic_pc = core.ps2pc(tonic_ps)
	mode_index = mode_candidates.index(mode)
	return num_mode * tonic_pc + mode_index

def key_index2symbol(key_index):
	tonic_pc, mode_index = parse_index(index)
	tonic_ps = core.pc2ps(tonic_pc)
	mode = mode_candidates[mode_index]
	return f"{root_pn}_{mode}"


