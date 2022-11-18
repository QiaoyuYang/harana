# Some basic elements to process a note and its pitch
# pc stands for pitch class and pn stands for pitch name

# Number of different pitch classes
num_pc = 12

# Number of different pitch spellings
num_natural_ps = 7
natural_ps_candidates = ["C", "D", "E", "F", "G", "A", "B"]
# Dictionary from natural pitch spelling to pitch class
natural_ps2pc = {
	'C' : 0,
	'D' : 2,
	'E' : 4,
	'F' : 5,
	'G' : 7,
	'A' : 9,
	'B' : 11
}

num_accidental = 5
accidental_candidates = ["bb", "b", "natural", "#", "##"]

# Dictionary from accidental to pc add
accidental2pc_add = {
	"bb" : -2,
	"b" : -1,
	"natural" : 0,
	"#" : 1,
	"##" : 2

}

num_ps = num_natural_ps * num_accidental


# Dictionary from pitch class to list of possible pitch spelling
pc2ps = {
	0 : ["B#", "C", "Dbb"],
	1 : ["B##", "C#", "Db"],
	2 : ["C##", "D", "Ebb"],
	3 : ["D#", "Eb", "Fbb"],
	4 : ["D##", "E", "Fb"],
	5 : ["E#", "F", "Gbb"],
	6 : ["E##", "F#", "Gb"],
	7 : ["F##", "G", "Abb"],
	8 : ["G#", "Ab"],
	9 : ["G##", "A", "Bbb"],
	10 : ["A#", "Bb", "Cbb"],
	11 : ["A##", "B", "Cb"]
}

num_quality = 10

# Candidate values for the quality
quality_candidates = ['maj', 'min', 'aug', 'dim', 'maj7', 'min7', 'dom7', 'hdi7', 'dim7', 'aug6']

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

class Note:

	def __init__(self, midi_num, duration, onset, offset):
		self.midi_num = midi_num
		self.duration = duration
		self.onset = onset
		self.offset = offset
		self.piano_pitch = midi_num2piano_pitch(midi_num)
		self.pc = midi_num2pc(midi_num)

	def __repr__(self):
		return f"Note(midi_num = {self.midi_num}, duration = {self.duration}, onset = {self.onset}, offset = {self.offset})"



def midi_num2piano_pitch(midi_num):
	return midi_num - 20

def piano_pitch2pc(piano_pitch):
	return (piano_pitch - 4)%12

def midi_num2pc(midi_num):
	return piano_pitch2pc(midi_num2piano_pitch(midi_num))

def pc2ps_list(pc):
	return pc2ps[pc]

def ps2pc(ps):
	natural_ps, accidental = parse_ps(ps)
	return (natural_ps2pc[natural_ps] + accidental2pc_add[accidental]) % num_pc

def parse_ps(ps):
	if len(ps) > 1:
		natural_ps = ps[0]
		accidental = ps[1:]
		return natural_ps, accidental
	else:
		natural_ps = ps
		accidental = "natural"
		return natural_ps, accidental


def ps2ps_index(ps):
	natural_ps, accidental = parse_ps(ps)
	return natural_ps_candidates.index(natural_ps) * num_accidental + accidental_candidates.index(accidental)

def ps_index2ps(ps_index):
	accidental_index = ps_index % num_accidental
	natural_ps_index = int((ps_index - accidental_index) / num_accidental)
	return natural_ps_candidates[natural_ps_index] + accidental_candidates[accidental_index]

def degree2pc_add(degree, mode):
	
	primary_degree, secondary_degree = parse_degree(degree)
	
	if mode == "maj":
		primary_degree_pc_add = degree2pc_add_maj[primary_degree]	
		secondary_degree_pc_add = degree2pc_add_maj[secondary_degree]
	
	if mode == "min":
		primary_degree_pc_add = degree2pc_add_min[primary_degree]
		secondary_degree_pc_add = degree2pc_add_min[secondary_degree]

	return (primary_degree_pc_add + secondary_degree_pc_add) % core.num_pc




