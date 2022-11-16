# Some basic classes and methods to process chords
# pc stands for pitch class and pn stands for pitch name

# My import
from . import core

# Regular import
import numpy as np

num_quality = 10

# Candidate values for root pitch class, quality and bass pitch class
root_pc_candidates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
quality_candidates = ['maj', 'min', 'aug', 'dim', 'maj7', 'min7', 'dom7', 'hdi7', 'dim7', 'aug6']
bass_pc_candidates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Dictionary from quality to the quality index
quality2quality_index = {
	'maj' : 0,
	'min' : 1,
	'aug' : 2,
	'dim' : 3,
	'maj7' : 4,
	'min7' : 5, 
	'dom7' : 6, 
	'dim7' : 7, 
	'hdi7' : 8,
	'aug6' : 9,
	}


# Dictionary from the chord quality to chordal degrees
quality2chordal_degree = {
	'maj' : [0,4,7],
	'min' : [0,3,7],
	'aug' : [0,4,8],
	'dim' : [0,3,6],
	'maj7' : [0,4,7,11],
	'min7' : [0,3,7,10], 
	'dom7' : [0,4,7,10],
	'hdi7' : [0,3,6,10],
	'dim7' : [0,3,6,9],
	'aug6' : [0,2,6],
}

# Dictionary from the chord quality to prior weight
quality_prior = {
	'maj' : 1,
	'min' : 1,
	'aug' : 0.1,
	'dim' : 0.5,
	'maj7' : 0.5,
	'min7' : 0.5, 
	'dom7' : 1, 
	'hdi7' : 0.5,
	'dim7' : 0.5,
	'aug6' : 0.1,
}

# Dictionary from degree to pitch class distance from the root
degree2pc_add_Maj = {
	'1' : 0,
	'-2' : 1,
	'2' : 2,
	'3' : 4,
	'4' : 5,
	'+4' : 6,
	'5' : 7,
	'6' : 9,
	'-7' : 10,
	'7' : 11
}

degree2pc_add_min = {
	'1' : 0,
	'-2' : 1,
	'2' : 2,
	'3' : 3,
	'4' : 5,
	'+4' : 6,
	'5' : 7,
	'6' : 8,
	'-7' : 10,
	'7' : 11
}

class Chord:

	def __init__(self, *args, **kwargs):
		arg_keys = list(kwargs.keys())
		self.onset = 0
		self.offset = 0
		self.duration = 0
		if "symbol" in arg_keys:
			self.symbol = kwargs["symbol"]
			self.root_pn, self.quality = parse_symbol(self.symbol)
			self.root_pc = core.pn2pc(self.root_pn)
			self.quality_index = quality2quality_index[self.quality]
		if "index" in arg_keys:
			self.index = kwargs["index"]
			self.root_pc, self.quality_index = parse_index(self.index)
			self.root_pn = core.pc2pn(self.root_pc)
			self.quality = quality_candidates[quality_index]
		if "root_pc" in arg_keys and "quality" in arg_keys:
			self.root_pc = parse_symbol(kwargs["root_pc"])
			self.root_pn = core.pc2pn(self.root_pn)
			self.quality = parse_symbol(kwargs["quality"])
			self.quality_index = quality2quality_index[self.quality]
		if "onset" in arg_keys and "offset" in arg_keys:
			self.onset = kwargs["onset"]
			self.offset = kwargs["offset"]
			self.duration = self.offset - self.onset + 1

	def __repr__(self):
		return f"Chord(symbol = {self.symbol}, boundary = [{self.onset}, {self.offset}])"

	def __str__(self):
		if self.onset == 0 and self.offset == 0:
			return self.symbol
		else:
			return f"({self.symbol}, {self.onset}, {self.offset})"

	def get_symbol(self):
		return f"{self.root_pn}_{self.quality}"

	def get_index(self):
		return self.root_pc * num_quality + self.quality_index


class Harmony(Chord):

	def __init__(self, *args, **kwargs):
		self.local_key = kwargs["local_key"]
		self.degree = kwargs["degree"]
		#print(self.degree)
		self.primary_degree, self.secondary_degree = parse_degree(self.degree)
		self.quality = kwargs["quality"]
		self.inversion = int(kwargs["inversion"])
		super().__init__(onset=kwargs["onset"], offset=kwargs["offset"])

		# Get the pitch class of root from the key and the degree
		if self.local_key.mode == "maj":
			self.root_pc = (self.local_key.root_pc + degree2pc_add_Maj[self.primary_degree] + degree2pc_add_Maj[self.secondary_degree])%12
		if self.local_key.mode == "min":
			self.root_pc = (self.local_key.root_pc + degree2pc_add_min[self.primary_degree] + degree2pc_add_Maj[self.secondary_degree])%12

		self.root_pn = core.pc2pn(self.root_pc)

		self.quality_index = quality2quality_index[self.quality]

		self.chordal_pc = get_chordal_pc(self.root_pc, self.quality)

		self.bass_pc = get_bass_pc(self.chordal_pc, self.inversion)

		self.symbol = self.get_symbol()

		self.index = self.get_index()

	def __repr__(self):
		return f"Harmony(key = {self.local_key}, degree = {self.degree}, quality = {self.quality}, inversion = {self.inversion}, boundary = [{self.onset}, {self.offset}])"

	def __str__(self):
		return f"({self.symbol}, {self.onset}, {self.offset})"


# Get the pitch classes of the chordal notes of a chord
def index2chordal_pc(index):
	root_pc, quality_index = parse_index(index)
	quality = quality_candidates[quality_index]
	return get_chordal_pc(root_pc, quality)

def get_chordal_pc(root_pc, quality):
	return [(root_pc + x) % 12 for x in quality2chordal_degree[quality]]

# Parse a chord symbol to find the root and the quality
def parse_symbol(symbol):
	return symbol.split('_')

# Parse a chord index to find the root and the quality
def parse_index(index):
	quality_index = index % num_quality
	root_pc = int((index - quality_index) / num_quality)
	return root_pc, quality_index

# Convert the symbol of a chord to its index
def symbol2index(symbol):
	root_pn, quality = parse_symbol(symbol)
	root_pc = core.pn2pc(root_pn)
	quality_index = quality2quality_index[quality]
	return num_quality * root_pc + quality_index

# Convert the index of chord to its symbol
def index2symbol(index):
	root_pc, quality_index = parse_index(index)
	root_pn = core.pc2pn(root_pc)
	quality = quality_candidates[quality_index]
	return f"{root_pn}_{quality}"

# Split the degree into the primary degree and the secondary degree
def parse_degree(degree):
	if "/" in degree:
		secondary_degree, primary_degree = degree.split("/")
	else:
		primary_degree = degree
		secondary_degree = "1"
	return primary_degree, secondary_degree

# Get the pitch class of the bass note of a chord depending on the inversion
def get_bass_pc(chordal_pc, inversion):
	if inversion == "0":
		return chordal_pc[0]
	if inversion == "1":
		return chordal_pc[1]
	if inversion == "2":
		return chordal_pc[2]
	if inversion == "3":
		return chordal_pc[3]












