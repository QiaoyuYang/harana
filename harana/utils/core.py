# Some basic elements to process a note and its pitch
# pc stands for pitch class and pn stands for pitch name

# Dictionary from pitch class to pitch name
pc2pn_dict = {
	0 : 'C',
	1 : 'C#',
	2 : 'D',
	3 : 'Eb',
	4 : 'E',
	5 : 'F',
	6 : 'F#',
	7 : 'G',
	8 : 'Ab',
	9 : 'A',
	10 : 'Bb',
	11 : 'B'
}

# Dictionary from pitch name to pitch class
pn2pc_dict = {
	'C' : 0,
	'C#' : 1,
	'Db' : 1,
	'D' : 2,
	'D#' : 3,
	'Eb' : 3,
	'E' : 4,
	'F' : 5,
	'F#' : 6,
	'Gb' : 6,
	'G' : 7,
	'G#' : 8,
	'Ab' : 8,
	'A' : 9,
	'A#' : 10,
	'Bb' : 10,
	'B' : 11
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

def pc2pn(pc):
	return pc2pn_dict[pc]

def pn2pc(pn):
	return pn2pc_dict[pn]