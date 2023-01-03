# Common codes to preprocess the datasets

# My import

from ..utils import core, harmony, chord, key, rn

# External import
import argparse
import os
import csv
import numpy as np
import pandas as pd
import torch


class HADataReader:
	# A generic data reader
	def __init__(self, dataset_root_dir, sample_size, dataset_name, tpqn, fpqn):
		
		# The root directory of all datasets
		self.dataset_root_dir = dataset_root_dir

		# The root directory of the current dataset
		self.dataset_dir = os.path.join(self.dataset_root_dir, dataset_name)

		self.sample_size = sample_size

		# Number of ticks per quarter notes
		self.tpqn = tpqn

		# Number of frames per quarter notes
		self.fpqn = fpqn
		
		# Number of pitch values on piano
		self.num_piano_pitch = 88
		
		# Number of distinctive pitch classes
		self.num_pc = 12
	
	def data2sample_tensors(self, frame_type):
		"""
		Convert the input data to tensors that could be more easily processed by Pytorch

		Parameters
		----------
		frame_type: string
			the type of basic time unit of the music
				possible values: "fixed_size",  "inter_onset"
				fixed_size: fixed frame length for all frames
				inter_onset: frame length given by the interval between adjacent distinct onsets of notes

		"""

		note_exist_frame_seq_all_samples = torch.tensor([])
		note_dur_frame_seq_all_samples = torch.tensor([])
		pc_exist_frame_seq_all_samples = torch.tensor([])
		pc_dur_frame_seq_all_samples = torch.tensor([])
		chord_frame_seq_all_samples = torch.tensor([])
		root_frame_seq_all_samples = torch.tensor([])
		quality_frame_seq_all_samples = torch.tensor([])
		key_frame_seq_all_samples = torch.tensor([])
		key_tonic_frame_seq_all_samples = torch.tensor([])
		key_mode_frame_seq_all_samples = torch.tensor([])
		pri_deg_frame_seq_all_samples = torch.tensor([])
		sec_deg_frame_seq_all_samples = torch.tensor([])
		inversion_frame_seq_all_samples = torch.tensor([])
		rn_frame_seq_all_samples = torch.tensor([])
		song_idx_all_samples = torch.tensor([])
		sample_idx_in_song_all_samples = torch.tensor([])
		qn_offset_cur_song_all_samples = torch.tensor([])
		
		# Enumerate through songs
		# During data reading, the song indexes are encoded as positive integers, starting from 1
		for song_idx in range(1, self.num_song + 1):
			
			print(f"Converting data to tensors: song {song_idx}")	
			
			# Get the notes of current song
			notes_cur_song = self.notes_all_song[song_idx]

			# Get the harmonies of current song
			harmonies_cur_song = self.harmonies_all_song[song_idx]

			# Get the quarter note offset of current song
			qn_offset = float(self.qn_offset_all_song[song_idx])
			
			# Check the frame type
			if frame_type == "inter_onset":
				print("Here")
				# Number of units per sample
				ups = 48

				# Find the onsets (possible split points for chord changes) and the notes within each inter-onset interval (unit)
				units_cur_song, split_points_cur_song = self.notes2units(notes_cur_song)

				# Convert the time units of chord boundaries to split point indexes
				self.chords_time2unit(split_points_cur_song, chords_cur_song)

				# Calculate the total number of frames
				total_frame = len(units_cur_song)

				# Calculate the number of samples
				num_sample = int(np.floor(total_frame / self.sample_size))

				# Initialize the tensors used to store data
				note_exist_seq, note_dist_seq, pc_exist_seq, pc_dist_seq, chord_seq, song_idx_cur_song, sample_idx_cur_song, qn_offset_cur_song = self.initialize_tensors(num_sample)

				# Auxillary assignment for the first frame
				sample_end_frame = 0
				for sample_idx in range(num_sample):
					# get the frame boundary of the current sample
					sample_start_frame = sample_end_frame
					sample_end_frame = sample_start_frame + self.sample_size

					chords_sample = self.get_chords_in_region(chords_cur_song, sample_start_frame, sample_end_frame)
					# Iterate through each frame in the current sample
					for frame_idx in range(self.sample_size):

						# Get the notes in the current frame
						notes_frame = units_cur_song[sample_start_frame + frame_idx]

						# Get the chords in the current frame
						chords_frame = self.get_chords_in_region(chords_sample, sample_start_frame + frame_idx, sample_start_frame + frame_idx + 1)
						
						note_exist_seq, note_dist_seq, pc_exist_seq, pc_dist_seq, chord_seq = self.update_tensors(note_exist_seq, note_dist_seq, pc_dist_seq, pc_exist_seq, chord_sequence, notes_frame, chords_frame, sample_idx, frame_idx)

			elif frame_type == "fixed_size":
				# Number of ticks per frame
				tpf = self.tpqn / self.fpqn

				# Number of ticks per sample
				tps = tpf * self.sample_size

				# Calculate the total number of ticks
				total_ticks = notes_cur_song[-1].offset - notes_cur_song[0].onset

				ticks_offset = qn_offset * self.tpqn
				
				# Calculate the number of samples in the current song
				num_sample = int(np.floor((total_ticks - ticks_offset) / tps))

				# Initialize the tensors used to store data
				self.initialize_tensors(num_sample)

				# Auxillary assignment for the first frame
				sample_end_frame = -1

				# get the starting tick of the first frame
				global_start_tick = 0

				# Iterate through each sample
				for sample_idx in range(num_sample):
					self.song_idx_cur_song[sample_idx] = song_idx
					self.sample_idx_in_song[sample_idx] = sample_idx
					self.qn_offset_cur_song[sample_idx] = qn_offset
					
					# get the frame boundary of the current sample
					sample_start_frame = sample_end_frame + 1
					sample_end_frame = sample_start_frame + self.sample_size - 1

					# get the tick boundary of the current sample
					sample_start_tick = global_start_tick + sample_start_frame * tpf
					sample_end_tick = sample_start_tick + tps - 1
					# get the notes in the sample from all notes to reduce the search space of tick notes
					notes_sample = self.get_notes_in_region(notes_cur_song, sample_start_tick, sample_end_tick)
					harmonies_sample = self.get_harmonies_in_region(harmonies_cur_song, sample_start_tick, sample_end_tick)

					# Iterate through each frame in the current sample
					for frame_idx in range(self.sample_size):
						#print(f"frame_idx: {frame_idx}")
						# Get the tick boundary of the current frame
						frame_start_tick = sample_start_tick + frame_idx * tpf
						frame_end_tick = frame_start_tick + tpf - 1

						# Get the notes in the frame from the sample notes
						notes_frame = self.get_notes_in_region(notes_sample, frame_start_tick, frame_end_tick)
						#print(notes_frame)
						harmonies_frame = self.get_harmonies_in_region(harmonies_sample, frame_start_tick, frame_end_tick)

						self.update_tensors(notes_frame, harmonies_frame, sample_idx, frame_idx)

			# Normalize the acculative pc tensors
			#note_dur_frame_seq = note_dur_frame_seq / note_dist_seq.sum(-1).unsqueeze(-1)
			#pc_dur_frame_seq = pc_dur_frame_seq / pc_dist_seq.sum(-1).unsqueeze(-1)

			self.sample_tensors = []
			
			# Concatenate the samples
			note_exist_frame_seq_all_samples = torch.cat([note_exist_frame_seq_all_samples, self.note_exist_frame_seq], dim=0)
			self.sample_tensors.append(note_exist_frame_seq_all_samples)
			
			note_dur_frame_seq_all_samples = torch.cat([note_dur_frame_seq_all_samples, self.note_dur_frame_seq], dim=0)
			self.sample_tensors.append(note_dur_frame_seq_all_samples)
			
			pc_exist_frame_seq_all_samples = torch.cat([pc_exist_frame_seq_all_samples, self.pc_exist_frame_seq], dim=0)
			self.sample_tensors.append(pc_exist_frame_seq_all_samples)
			
			pc_dur_frame_seq_all_samples = torch.cat([pc_dur_frame_seq_all_samples, self.pc_dur_frame_seq], dim=0)
			self.sample_tensors.append(pc_dur_frame_seq_all_samples)
			
			chord_frame_seq_all_samples = torch.cat([chord_frame_seq_all_samples, self.chord_frame_seq], dim=0)
			self.sample_tensors.append(chord_frame_seq_all_samples)

			root_frame_seq_all_samples = torch.cat([root_frame_seq_all_samples, self.root_frame_seq], dim=0)
			self.sample_tensors.append(root_frame_seq_all_samples)

			quality_frame_seq_all_samples = torch.cat([quality_frame_seq_all_samples, self.quality_frame_seq], dim=0)
			self.sample_tensors.append(quality_frame_seq_all_samples)

			key_frame_seq_all_samples = torch.cat([key_frame_seq_all_samples, self.key_frame_seq], dim=0)
			self.sample_tensors.append(key_frame_seq_all_samples)

			key_tonic_frame_seq_all_samples = torch.cat([key_tonic_frame_seq_all_samples, self.key_tonic_frame_seq], dim=0)
			self.sample_tensors.append(key_tonic_frame_seq_all_samples)

			key_mode_frame_seq_all_samples = torch.cat([key_mode_frame_seq_all_samples, self.key_mode_frame_seq], dim=0)
			self.sample_tensors.append(key_mode_frame_seq_all_samples)

			rn_frame_seq_all_samples = torch.cat([rn_frame_seq_all_samples, self.rn_frame_seq], dim=0)
			self.sample_tensors.append(rn_frame_seq_all_samples)
			
			pri_deg_frame_seq_all_samples = torch.cat([pri_deg_frame_seq_all_samples, self.pri_deg_frame_seq], dim=0)
			self.sample_tensors.append(pri_deg_frame_seq_all_samples)

			sec_deg_frame_seq_all_samples = torch.cat([sec_deg_frame_seq_all_samples, self.sec_deg_frame_seq], dim=0)
			self.sample_tensors.append(sec_deg_frame_seq_all_samples)

			inversion_frame_seq_all_samples = torch.cat([inversion_frame_seq_all_samples, self.inversion_frame_seq], dim=0)
			self.sample_tensors.append(inversion_frame_seq_all_samples)

			song_idx_all_samples = torch.cat([song_idx_all_samples, self.song_idx_cur_song], dim=0)
			self.sample_tensors.append(song_idx_all_samples)

			sample_idx_in_song_all_samples = torch.cat([sample_idx_in_song_all_samples, self.sample_idx_in_song], dim=0)
			self.sample_tensors.append(sample_idx_in_song_all_samples)

			qn_offset_cur_song_all_samples = torch.cat([qn_offset_cur_song_all_samples, self.qn_offset_cur_song], dim=0)
			self.sample_tensors.append(qn_offset_cur_song_all_samples)

	# Initialize the tensors to store the ground truth data
	def initialize_tensors(self, num_sample):

		# 88-dimensional one_hot vector for piano pitch appearance in each frame
		self.note_exist_frame_seq = torch.zeros(num_sample, self.sample_size, self.num_piano_pitch)

		# 88-dimensional vector for piano pitch distribution in each frame
		self.note_dur_frame_seq = torch.zeros(num_sample, self.sample_size, self.num_piano_pitch)

		# 12-dimensional one_hot vector for pitch class appearance in each frame
		self.pc_exist_frame_seq = torch.zeros(num_sample, self.sample_size, self.num_pc)

		# 12-dimensional vector for pitch class distribution in each frame
		self.pc_dur_frame_seq = torch.zeros(num_sample, self.sample_size, self.num_pc)

		# one chord label for each frame
		self.chord_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one root label for each frame
		self.root_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one quality label for each frame
		self.quality_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one key label for each frame
		self.key_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one key tonic label for each frame
		self.key_tonic_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one key mode label for each frame
		self.key_mode_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one primary degree label for each frame
		self.pd_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one secondary degree label for each frame
		self.sd_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one inversion label for each frame
		self.inversion_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one rn label for each frame
		self.rn_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one primary degree label for each frame
		self.pri_deg_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one secondary degree label for each frame
		self.sec_deg_frame_seq = torch.zeros(num_sample, self.sample_size)

		# one inversion label for each frame
		self.inversion_frame_seq = torch.zeros(num_sample, self.sample_size)

		# the song index of the each sample
		self.song_idx_cur_song = torch.zeros(num_sample)

		# the sample index in the song of the each sample
		self.sample_idx_in_song = torch.zeros(num_sample)

		# the quarter note offset of the song of the each sample
		self.qn_offset_cur_song = torch.zeros(num_sample)

	# Update the tensors as new samples are created
	def update_tensors(self, notes_frame, harmonies_frame, sample_idx, frame_idx):
		
		# Iterate through each note in the current frame
		for note_cur in notes_frame:

			# Update the feature tensors related to notes and pitch classes
			if self.note_exist_frame_seq[sample_idx, frame_idx, note_cur.piano_pitch - 1] == 0:
				self.note_exist_frame_seq[sample_idx, frame_idx, note_cur.piano_pitch - 1] = 1
			self.note_dur_frame_seq[sample_idx, frame_idx, note_cur.piano_pitch - 1] += note_cur.duration
			if self.pc_exist_frame_seq[sample_idx, frame_idx, note_cur.pc] == 0:
				self.pc_exist_frame_seq[sample_idx, frame_idx, note_cur.pc] = 1
			self.pc_dur_frame_seq[sample_idx, frame_idx, note_cur.pc] += note_cur.duration

		# Update the feature tensors related to chords
		harmony_frame = harmonies_frame[0]
		self.chord_frame_seq[sample_idx, frame_idx] = harmony_frame.this_chord.chord_index
		self.root_frame_seq[sample_idx, frame_idx] = harmony_frame.this_chord.root_pc
		self.quality_frame_seq[sample_idx, frame_idx] = harmony_frame.this_chord.quality_index
		self.key_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.local_key.key_index
		self.key_tonic_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.local_key.tonic_pc
		self.key_mode_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.local_key.mode_index
		self.rn_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.rn_index
		self.pri_deg_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.primary_degree_index
		self.sec_deg_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.secondary_degree_index
		self.inversion_frame_seq[sample_idx, frame_idx] = harmony_frame.roman_numeral.inversion

	# Split all the notes in a song into inter-onset regions (unit)
	def notes2units(self, notes_all):
		
		# Find the first and last split point: first onset and last offset
		first_onset = notes_all[-1].onset
		last_offset = notes_all[0].offset
		for note_cur in notes_all:
			if note_cur.onset < first_onset:
				first_onset = note_cur.onset
			if note_cur.offset > last_offset:
				last_offset = note_cur.offset

		# All split points are onsets of a note except the last one
		split_points = [first_onset]
		for note_cur in notes_all:
			if note_cur.onset > split_points[-1]:
				split_points.append(note_cur.onset)
		
		split_points.append(last_offset)
		
		# Get the notes (or partial notes) within each unit
		units_all = []
		for idx in range(len(split_points) - 1):
			start_time = split_points[idx]
			end_time = split_points[idx + 1]
			notes_unit = self.get_notes_in_region(notes_all, start_time, end_time)
			units_all.append(notes_unit)

		return units_all, split_points

	# Convert the time units of the chord boundaries to split points
	def chords_time2unit(self, split_points, chords_all):

		for chord_cur in chords_all:
			chord_onset = chord_cur.onset
			chord_offset = chord_cur.offset
			for i in range(len(split_points)):
				if chord_onset == split_points[i]:
					chord_onset_unit_idx = i
				if chord_offset == split_points[i]:
					chord_offset_unit_idx = i
			if chord_offset_unit_idx < chord_onset_unit_idx:
				chord_offset_unit_idx = chord_onset_unit_idx

			chord_cur.onset = chord_onset_unit_idx
			chord_cur.offset = chord_offset_unit_idx

	# Get the (partial) notes in a time frame
	def get_notes_in_region(self, notes_all, start_time, end_time):
		
		notes_region = []
		for note_cur in notes_all:
			note_cur_onset = note_cur.onset
			note_cur_offset = note_cur.offset
			if self.check_duration_overlap(note_cur_onset, note_cur_offset, start_time, end_time):
				start_time_overlap, end_time_overlap, duration_overlap = self.get_duration_overlap(note_cur.onset, note_cur.offset, start_time, end_time)
				notes_region.append(core.Note(note_cur.midi_num, duration_overlap, start_time_overlap, end_time_overlap))
		return notes_region

	# Get the (partial) harmonies in a time frame
	def get_harmonies_in_region(self, harmonies_all, start_time, end_time):
		
		harmonies_region = []
		for harmony_cur in harmonies_all:
			harmony_cur_onset = harmony_cur.onset
			harmony_cur_offset = harmony_cur.offset
			if self.check_duration_overlap(harmony_cur_onset, harmony_cur_offset, start_time, end_time):
				start_time_overlap, end_time_overlap, duration_overlap = self.get_duration_overlap(harmony_cur.onset, harmony_cur.offset, start_time, end_time)
				harmonies_region.append(harmony.Harmony(local_key=harmony_cur.local_key, degree=harmony_cur.degree, quality=harmony_cur.quality, inversion=harmony_cur.inversion, onset=start_time_overlap, offset=end_time_overlap))
		
		return harmonies_region

	# Check if the time span of a event overlap with another given time span
	def check_duration_overlap(self, event_onset, event_offset, start_time, end_time):

		return (event_onset >= start_time and event_offset <= end_time) or (event_onset < start_time and event_offset > start_time) or (event_onset < end_time and event_offset > end_time)
	
	# Get the overlapping duration of two overlappinng time spans
	def get_duration_overlap(self, event_onset, event_offset, start_time, end_time):
		
		if event_onset < start_time:
			if event_offset < end_time:
				return start_time, event_offset, event_offset - start_time + 1
			else:
				return start_time, end_time, end_time - start_time + 1
		else:
			if event_offset < end_time:
				return event_onset, event_offset, event_offset - event_onset + 1
			else:
				return event_onset, end_time, end_time - event_onset + 1




