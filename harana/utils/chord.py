# Some basic classes and methods to process chords
# pc stands for pitch class and pn stands for pitch name

# My import
from . import core

# Regular import
import numpy as np

# Candidate values for root pitch class, quality and bass pitch class
root_pc_candidates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
quality_candidates = ['M', 'm', '+', '-', 'Maj7', 'min7', 'Dom7', 'dim7', 'hdi7', 'It+6', 'Fr+6', 'Gr+6']
bass_pc_candidates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Dictionary from the chord quality to chordal degrees
quality2chordal_degree_dict = {
	'M' : [0,4,7],
	'm' : [0,3,7],
	'+' : [0,4,8],
	'-' : [0,3,6],
	'Maj7' : [0,4,7,11],
	'min7' : [0,3,7,10], 
	'Dom7' : [0,4,7,10], 
	'dim7' : [0,3,6,10], 
	'hdi7' : [0,3,6,9],
	'It+6' : [0,2,6],
	'Fr+6' : [0,2,6,8],
	'Gr+6' : [0,2,6,9],
}

# Dictionary from the chord quality to prior weight
quality_prior = {
	'M' : 1,
	'm' : 1,
	'+' : 0.1,
	'-' : 0.5,
	'Maj7' : 0.5,
	'min7' : 0.5, 
	'Dom7' : 1, 
	'dim7' : 0.5, 
	'hdi7' : 0.5,
	'It+6' : 0.1,
	'Fr+6' : 0.1,
	'Gr+6' : 0.1,
}

# Dictionary from degree to pitch class distance from the root
degree2pc_add_dict = {
	'1' : 0,
	'+1' : 1,
	'-2' : 1,
	'2' : 2,
	'+2' : 3,
	'-3' : 3,
	'3' : 4,
	'4' : 5,
	'+4' : 6,
	'-5' : 6,
	'5' : 7,
	'+5' : 8,
	'-6' : 8,
	'6' : 9,
	'+6' : 10,
	'-7' : 10,
	'7' : 11
}

class Chord:

	def __init__(self, root_pc, quality, bass_pc, duration, onset, offset):

		self.root_pc = root_pc
		self.root_pn = core.pc2pn(self.root_pc)
		self.quality = quality
		self.bass_pc = bass_pc
		self.bass_pn = core.pc2pn(self.bass_pc)
		self.onset = onset
		self.offset = offset
		self.duration = duration
		self.symbol = self.get_symbol()
		# a distinct integer number to represent each chord label
		self.index = self.get_index()


	def __repr__(self):
		if self.root_pn == self.bass_pn:
			return f"Chord({self.root_pn}_{self.quality}, duration = {self.duration}, boundary=[{self.onset}, {self.offset}])"
		else:
			return f"Chord(label={self.root_pn}_{self.quality}/{self.bass_pn}, duration = {self.duration}, boundary=[{self.onset}, {self.offset}])"

	def __str__(self):
		if self.root_pn == self.bass_pn:
			return f"Chord(label={self.root_pn}_{self.quality}, duration = {self.duration}, boundary=[{self.onset}, {self.offset}])"
		else:
			return f"Chord(label={self.root_pn}_{self.quality}/{self.bass_pn}, duration = {self.duration}, boundary=[{self.onset}, {self.offset}])"

	def get_symbol(self):
		if self.root_pn == self.bass_pn:
			return f"{self.root_pn}_{self.quality}"
		else:
			return f"{self.root_pn}_{self.quality}/{self.bass_pn}"

	def get_index(self):

		return symbol2index(self.symbol)

	def __eq__(self, other):
		if not isinstance(other, Chord):
			return False
		else:
			return self.quality == other.quality and self.root_pc == other.root_pc and self.bass_pc == other.bass_pc

# Get the pitch classes of the chordal notes of a chord
def index2chordal_pc(index):
	symbol = index2symbol(index)
	root_pc, quality, _ = parse_symbol(symbol)
	return get_chordal_pc(root_pc, quality)

def get_chordal_pc(root_pc, quality):
	return [(root_pc + x)%12 for x in quality2chordal_degree_dict[quality]]

# Get all the chordal notes on a piano of a chord
def get_chordal_notes(root_pc, quality):
	chordal_degree = quality2chordal_degree_dict[quality]
	chordal_notes = []
	for piano_pitch in range(1, 89):
		if (piano_pitch - (root_pc + 4))%12 in chordal_degree:
			chordal_notes.append(piano_pitch)
	return chordal_notes

# Parse a chord symbol to find the root, quality and the bass
def parse_symbol(symbol):

	# check if the bass is different from the root
	if '/' not in symbol:
		root_pn, quality = symbol.split('_')
		bass_pn = root_pn
	else:
		symbol_main, bass_pn = symbol.split('/')
		root_pn, quality = symbol_main.split('_')

	return core.pn2pc(root_pn), quality, core.pn2pc(bass_pn)

# Convert the symbol of a chord to its index
def symbol2index(symbol):
	root_pc, quality, bass_pc = parse_symbol(symbol)

	root_idx = root_pc_candidates.index(root_pc)
	quality_idx = quality_candidates.index(quality)
	bass_idx = bass_pc_candidates.index(bass_pc)

	#return root_idx * 144 + quality_idx * 12 + bass_idx

	return root_idx * 12 + quality_idx

# Convert the index of chord to its symbol
def index2symbol(index):
	
	'''
	root_idx = int(np.floor(index / 144))
	index_remain = index - root_idx * 144
	quality_idx = int(np.floor(index_remain / 12))
	bass_idx = index_remain - quality_idx * 12

	root_pc = root_pc_candidates[root_idx]
	quality = quality_candidates[quality_idx]
	bass_pc = bass_pc_candidates[bass_idx]

	if root_pc == bass_pc:
		return f"{core.pc2pn(root_pc)}_{quality}"
	else:
		return f"{core.pc2pn(root_pc)}_{quality}/{core.pc2pn(bass_pc)}"
	'''

	root_idx = int(np.floor(index / 12))
	quality_idx = index - root_idx * 12
	root_pc = root_pc_candidates[root_idx]
	quality = quality_candidates[quality_idx]
	
	return f"{core.pc2pn(root_pc)}_{quality}"









