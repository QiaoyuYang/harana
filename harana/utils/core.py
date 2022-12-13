# Some basic elements to process a note and its pitch
# pc stands for pitch class and pn stands for pitch name

# Number of different pitch classes
num_pc = 12

# Dictionary from pitch class to one possible pitch spelling
# There could be multiple options for each pitch class. We choose a commonly used one.
pc2ps_dict = {
	0 : "C",
	1 : "C#",
	2 : "D",
	3 : "Eb",
	4 : "E",
	5 : "F",
	6 : "F#",
	7 : "G",
	8 : "Ab",
	9 : "A",
	10 : "Bb",
	11 : "B"
}

ps2pc_dict = {
	"B#" : 0,
	"C" : 0,
	"C#" : 1,
	"Db" : 1,
	"D" : 2,
	"D#" : 3,
	"Eb" : 3,
	"E" : 4,
	"Fb" : 4,
	"E#" : 5,
	"F" : 5,
	"F#" : 6,
	"Gb" : 6,
	"G" : 7,
	"G#" : 8,
	"Ab" : 8,
	"A" : 9,
	"A#" : 10,
	"Bb" : 10,
	"B" : 11
}

num_quality = 10

# Candidate values for the quality
quality_candidates = ['maj', 'min', 'aug', 'dim', 'maj7', 'min7', 'dom7', 'hdi7', 'dim7', 'aug6']

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
	return (piano_pitch - 4) % 12

def midi_num2pc(midi_num):
	return (midi_num - 24) % 12

def pc2ps(pc):
	return pc2ps_dict[pc]

def ps2pc(ps):
	return ps2pc_dict[ps]


