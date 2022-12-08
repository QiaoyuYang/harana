from ..utils import core

# Number of modes used in the model
num_mode = 2
mode_candidates = ["maj", "min"]

num_single_accidental = 3
single_accidental_candidates = ["b", "natural", "#"]

# Number of different pitch spellings for key tonics
num_tonic_ps = core.num_natural_ps * num_single_accidental
num_key = num_tonic_ps * num_mode

tonic_ps_candidates = [
    "Cb",
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "E#",
    "Fb",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
    "B#",
]

class Key:

	def __init__(self, *args, **kwargs):

		self.key_index = None
		self.key_symbol = None
		arg_keys = list(kwargs.keys()) 
		if arg_keys == ["tonic_ps", "mode"]:
			self.tonic_ps = kwargs["tonic_ps"]
			self.tonic_index = tonic_ps2index(self.tonic_ps)
			self.mode = kwargs["mode"]
			self.mode_index = mode_candidates.index(self.mode)
		elif arg_keys == ["key_symbol"]:
			self.key_symbol = kwargs["key_symbol"]
			self.tonic_ps, self.mode = parse_key_symbol(self.key_symbol)
			self.tonic_index = tonic_ps2index(self.tonic_ps)
			self.mode_index = mode_candidates.index(self.mode)
		elif arg_keys == ["key_index"]:
			self.key_index = kwargs["key_index"]
			self.tonic_index, self.mode_index = parse_key_index(self.key_index)
			self.tonic_ps = tonic_index2ps(self.tonic_index)
			self.mode = mode_candidates[self.mode_index]

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
		return num_mode * self.tonic_index + self.mode_index
	
def parse_key_symbol(key_symbol):
	tonic_ps, mode = key_symbol.split("_")
	return tonic_ps, mode

def parse_key_index(key_index):
	mode_index = key_index % num_mode
	tonic_index = int((index - mode_index) / num_mode)
	return tonic_index, mode_index

def key_symbol2index(key_symbol):
	tonic_ps, mode = parse_symbol(symbol)
	tonic_index = tonic_ps2index(root_pn)
	mode_index = mode_candidates.index(mode)
	return num_mode * tonic_index + mode_index

def key_index2symbol(key_index):
	tonic_index, mode_index = parse_index(index)
	tonic_ps = tonic_index2ps(root_pc)
	mode = mode_candidates[mode_index]
	return f"{root_pn}_{mode}"

def tonic_ps2index(tonic_ps):
	natural_ps, accidental = core.parse_ps(tonic_ps)
	return core.natural_ps_candidates.index(natural_ps) * num_single_accidental + single_accidental_candidates.index(accidental)

def tonic_index2ps(tonic_index):
	single_accidental_index = tonic_index % num_single_accidental
	natural_ps_index = int((tonic_index - single_accidental_index) / num_single_accidental)
	return core.natural_ps_candidates[natural_ps_index] + single_accidental_candidates[single_accidental_index]


