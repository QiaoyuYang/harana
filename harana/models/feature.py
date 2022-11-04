from ..utils import chord
from utils import core


class FrameFeatureExtractor:
	def __init__(self, data, batch_size, sample_size, num_label):
		self.num_pitch_class = 12
		self.num_piano_pitch = 88
		self.data = data
		self.batch_size = batch_size
		self.sample_size = sample_size

		self.num_label = 144
		# 88_dimensional: chordal_notes_appearing, chordal_notes_missing, non_chordal_notes
		# 12_dimensional: chordal_pc_appearing, chordal_pc_accumulation , chordal_pc_missing, non_chordal_pc
		self.num_feature = self.num_pitch_class * 3 + self.num_pitch_class * 4
		self.features = torch.zeros(batch_size, sample_size, num_labels, num_features)

	def extract_all(self):
		chordal_pc_appearing = torch.zeros(num_sample, sample_size, self.num_pitch_class)
		chordal_pc_accumulation = torch.zeros(num_sample, sample_size, self.num_pitch_class)
		chorda_pc_missing = torch.zeros(num_sample, sample_size, self.num_pitch_class)
		non_chordal_pc = torch.zeros(num_sample, sample_size, self.num_pitch_class)
		for label_idx in range(num_label):
			chordal_notes_cur = chord.get_chordal_notes(chord_cur.root_pc, chord_cur.quality)
			chordal_pc_cur = chord.get_chordal_pc(chord_cur.root_pc, chord_cur.quality)
			for chordal_note_cur in chordal_notes_cur:
				chordal_pc[sample_idx, frame_idx, core.piano_pitch2pc(chordal_note_cur)] = 1
				chordal_pc_accumulation[sample_idx, frame_idx, core.piano_pitch2pc(chordal_note_cur)] += 1
			self.features[:, :, label_idx, ] = self.extract(start_frame, end_frame, label)

	# To extract the features within segment
	def extract_segment(self, start_frame, end_frame, label):
		piano_roll = self.data['piano roll']
		for i in range(start_frame, end_frame + 1):
			self.features_in_segment = self_data[:, ]

