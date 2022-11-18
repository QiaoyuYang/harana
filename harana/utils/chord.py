# Some basic classes and methods to process chords
# pc stands for pitch class and pn stands for pitch name

# My import
from . import core

# Regular import
import numpy as np


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

class Chord:

	def __init__(self, *args, **kwargs):
		arg_keys = list(kwargs.keys())

		self.chord_symbol = None
		self.chord_index = None
		self.onset = 0
		self.offset = 0
		self.inversion = "0"
		if "chord_symbol" in arg_keys:
			self.chord_symbol = kwargs["chord_symbol"]
			self.root_ps, self.quality = parse_chord_symbol(self.chord_symbol)
			self.root_pc = core.ps2pc(self.root_ps)
			self.quality_index = core.quality_candidates.index(self.quality)
		if "chord_index" in arg_keys:
			self.chord_index = kwargs["chord_index"]
			self.root_pc, self.quality_index = parse_index(self.chord_index)
			self.root_ps = core.pc2ps_list(self.root_pc)[1]
			self.quality = core.quality_candidates[quality_index]
		if "root_pc" in arg_keys and "quality" in arg_keys:
			self.root_pc = kwargs["root_pc"]
			self.root_ps = core.pc2ps_list(self.root_pc)[1]
			self.quality = kwargs["quality"]
			self.quality_index = core.quality_candidates.index(self.quality)
		if "onset" in arg_keys and "offset" in arg_keys:
			self.onset = kwargs["onset"]
			self.offset = kwargs["offset"]
			self.duration = self.offset - self.onset + 1
		if "inversion" in arg_keys:
			self.inversion = kwargs["inversion"]

		self.chordal_pc = get_chordal_pc(self.root_pc, self.quality)
		self.bass_pc = get_bass_pc(self.chordal_pc, self.inversion)

		if not self.chord_symbol:
			self.chord_symbol = self.get_symbol()
		if not self.chord_index:
			self.chord_index = self.get_index()

	def __repr__(self):
		return f"Chord(symbol = {self.chord_symbol}, boundary = [{self.onset}, {self.offset}])"

	def __str__(self):
		if self.onset == 0 and self.offset == 0:
			return self.chord_symbol
		else:
			return f"({self.chord_symbol}, {self.onset}, {self.offset})"

	def get_symbol(self):
		return f"{self.root_ps}_{self.quality}"

	def get_index(self):
		return self.root_pc * core.num_quality + self.quality_index


# Get the pitch classes of the chordal notes of a chord
def chord_index2chordal_pc(chord_index):
	root_pc, quality_index = parse_index(index)
	quality = harmony.quality_candidates[quality_index]
	return get_chordal_pc(root_pc, quality)

def get_chordal_pc(root_pc, quality):
	return [(root_pc + x) % 12 for x in quality2chordal_degree[quality]]

# Parse a chord symbol to find the root and the quality
def parse_chord_symbol(chord_symbol):
	root_ps, quality = chord_symbol.split("_")
	return root_ps, quality

# Parse a chord index to find the root and the quality
def parse_chord_index(chord_index):
	quality_index = chord_index % core.num_quality
	root_pc = int((chord_index - quality_index) / core.num_quality)
	return root_pc, quality_index

# Convert the symbol of a chord to its index
def chord_symbol2index(chord_symbol):
	root_ps, quality = parse_chord_symbol(chord_symbol)
	root_pc = core.ps2pc(root_ps)
	quality_index = harmony.quality_candidates.index(quality)
	return core.num_quality * root_pc + quality_index

# Convert the index of chord to its symbol
def chord_index2symbol(chord_index):
	root_pc, quality_index = parse_index(chord_index)
	root_ps = core.pc2ps(root_pc)[1]
	quality = core.quality_candidates[quality_index]
	return f"{root_ps}_{quality}"

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