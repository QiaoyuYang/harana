from ..utils import core

# Number of modes used in the model
num_mode = 2

# Dictionary from mode to mode index
mode2mode_index_dict = {
	"Maj" : 0,
	"min" : 1
}

# Dictionary from mode index to mode
mode_index2mode_dict = {
	0 : "Maj",
	1 : "min"
}

class Key:

	def __init__(self, *args, **kwargs):
		arg_keys = list(kwargs.keys()) 
		if arg_keys == ["root_pn", "mode"]:
			self.root_pn = kwargs["root_pn"]
			self.root_pc = core.pn2pc(self.root_pn)
			self.mode = mode
			self.mode_index = mode2mode_index(self.mode)
		elif arg_keys == ["symbol"]:
			self.root_pn, self.mode = parse_symbol(kwargs["symbol"])
			self.root_pc = core.pn2pc(self.root_pn)
			self.mode_index = mode2mode_index(self.mode)
		elif arg_keys == ["index"]:
			self.root_pc, self.mode_index = parse_index(kwargs['index'])
			self.root_pn = core.pc2pn(self.root_pc)
			self.mode = mode_index2mode(self.mode_index)

	def __repr__(self):
		return f"Key(root = {self.root_pn}, mode = {self.mode})"
	
	def __str__(self):
		return self.get_symbol()
	
	def get_symbol(self):
		return f"{self.root_pn}_{self.mode}"

	def get_index(self):
		return num_mode * self.root_pc + self.mode_index

def mode2mode_index(mode):
	return mode2mode_index_dict[mode]

def mode_index2mode(mode_index):
	return mode_index2mode_dict[mode_index]
	
def parse_symbol(symbol):
	return symbol.split("_")

def parse_index(index):
	root_pc = index % num_mode
	mode_index = index - num_mode * root_pc
	return root_pc, mode_index


def symbol2index(symbol):
	root_pn, mode = parse_symbol(symbol)
	root_pc = core.pn2pc(root_pn)
	mode_index = mode2mode_index(mode)
	return mode_index * num_pc + root_pc

def index2symbol(index):
	root_pc, mode_index = parse_index(index)
	root_pn = pn2pc(root_pc)
	mode = mode_index2mode(mode_index)
	return f"{root_pn}_{mode}"


