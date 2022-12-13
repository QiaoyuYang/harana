# A harmony wrapper that support both chords and roman numerals

# My import
from . import core, chord, rn

# Dictionary from degree to pitch class distance from the root
degree2pc_add_maj = {
	"1" : 0,
	"-2" : 1,
	"2" : 2,
	"3" : 4,
	"4" : 5,
	"+4" : 6,
	"5" : 7,
	"6" : 9,
	"-7" : 10,
	"7" : 11
}

degree2pc_add_min = {
	"1" : 0,
	"-2" : 1,
	"2" : 2,
	"3" : 3,
	"4" : 5,
	"+4" : 6,
	"5" : 7,
	"6" : 8,
	"-7" : 10,
	"7" : 11
}

num_inversion = 4

class Harmony:

	def __init__(self, local_key, degree, quality, inversion, onset, offset):
		
		self.local_key = local_key
		self.degree = degree
		self.quality = quality
		self.inversion = inversion
		key_tonic_ps = self.local_key.tonic_ps
		mode = self.local_key.mode
		key_tonic_pc = core.ps2pc(key_tonic_ps)
		degree_pc_add = degree2pc_add(degree, mode)
		root_pc = (key_tonic_pc + degree_pc_add) % core.num_pc
		

		self.onset = onset
		self.offset = offset
		
		self.roman_numeral = rn.RomanNumeral(local_key = local_key, degree = degree, quality = quality, inversion = inversion, onset = onset, offset = offset)
		self.this_chord = chord.Chord(root_pc = root_pc, quality = quality, inversion = inversion, onset = onset, offset = offset)

	def __repr__(self):
		return f"Harmony(roman_numeral = {self.roman_numeral}, chord = {self.this_chord}, boundary = [{self.onset}, {self.offset}])"

	def __str__(self):
		return f"Harmony(roman_numeral = {self.roman_numeral}, chord = {self.this_chord}, boundary = [{self.onset}, {self.offset}])"

def degree2pc_add(degree, mode):
	
	primary_degree, secondary_degree = rn.parse_degree(degree)
	
	if mode == "maj":
		primary_degree_pc_add = degree2pc_add_maj[primary_degree]	
		secondary_degree_pc_add = degree2pc_add_maj[secondary_degree]
	
	if mode == "min":
		primary_degree_pc_add = degree2pc_add_min[primary_degree]
		secondary_degree_pc_add = degree2pc_add_min[secondary_degree]

	return (primary_degree_pc_add + secondary_degree_pc_add) % core.num_pc







