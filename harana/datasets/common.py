# Common codes to preprocess the datasets

# My import

from .. import tools

# External import
import os
import math
import shutil
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from abc import abstractmethod
import torch
from torch.utils.data import Dataset

class HaranaDataset(Dataset):
	
	# A generic class for harmony datasets
	def __init__(self, base_dir, splits, sample_size, harmony_type, reset_data, store_data, save_data, save_loc, key_tonic_aug, key_mode_aug, beat_as_unit, validation, seed):

		# Select a default base directory path if none was provided
		self.base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, self.dataset_name()) if base_dir is None else base_dir
		# Check if the dataset exists (in full) at the specified path
		if not os.path.exists(self.base_dir) or len(os.listdir(self.base_dir)) != len(self.available_tracks()) + 1:
			warnings.warn(f'Dataset was incomplete or could not find at specified '
				f'path \'{self.base_dir}\'. Attempting to download...', category=RuntimeWarning)
			# Download the dataset if it is missing
			self.download(self.base_dir)

		self.key_tonic_aug = key_tonic_aug
		self.key_mode_aug = key_mode_aug
		self.beat_as_unit = beat_as_unit
		self.validation = validation

		# Choose all available dataset splits if none were provided
		self.splits = self.available_splits() if splits is None else splits

		self.sample_size = tools.DEFAULT_SAMPLE_SIZE if sample_size is None else sample_size

		self.harmony_type = tools.DEFAULT_HARMONY_TYPE if harmony_type is None else harmony_type 


		# Set the storing and saving parameters
		self.store_data = store_data
		self.save_data = save_data
		self.save_loc = tools.DEFAULT_GROUND_TRUTH_DIR if save_loc is None else save_loc

		self.tracks = list()
		# Aggregate all the track names from the selected splits
		for split in self.splits:
			self.tracks += self.get_tracks(split)


		self.init_time_stats()
		self.init_mode_dist_stats()
		if os.path.exists(self.get_gt_path()) and reset_data:
			# Remove any saved ground-truth for the dataset
			shutil.rmtree(self.get_gt_path())
			self.init_harmony_transitions()

		if self.save_data:
			# Make sure the directory for saving/loading ground-truth exists
			os.makedirs(self.get_gt_path(), exist_ok=True)

		if not reset_data:
			self.tracks = np.load(os.path.join(self.get_gt_path(), 'tracks.npy'))

		# Initialize a random number generator for the dataset
		self.rng = np.random.RandomState(seed)

		# Load the ground-truth for each track into RAM
		if self.store_data:
			# Initialize a dictionary to store track data
			self.data = dict()
			for track in tqdm(self.tracks):
				# Load the data for the track
				self.data[track] = self.load(track)

			harmony_transitions_path
			if reset_data:
				self.harmony_transitions
				if key_mode_aug:
					print('Augmenting the mode...')
					self.key_mode_augmentation()


		if self.save_data:
			np.save(os.path.join(self.get_gt_path(), 'tracks.npy'), self.tracks)


		#self.print_time_stats()
		#self.print_mode_dist_stats()

	def init_harmony_transitions(self):
		
		num_harmonies = tools.NUM_HARMONIES[self.harmony_type]
		self.harmony_transitions = np.zeros((num_harmonies, num_harmonies))

	def init_mode_dist_stats(self):
		self.mode_dist = np.zeros((self.__len__(), 2))

	def print_mode_dist_stats(self):
		self.mode_dist = np.stack([self.mode_dist[:, i] / self.mode_dist.sum(axis=1) for i in range(2)], axis=1)
		print('Mode distribution of each track')
		print('Major\t\tMinor')
		print(self.mode_dist)
		print('Overall Mode Distribution')
		mode_dist_overall = self.mode_dist.sum(axis=0)
		mode_dist_overall = mode_dist_overall/mode_dist_overall.sum(axis=0)
		print('Major\t\tMinor')
		print(mode_dist_overall)

		mode_dist_balanced = deepcopy(self.mode_dist)

		for i in range(self.__len__()):
			diff = max(0.5-mode_dist_overall[0], 0) * max(mode_dist_balanced[i, 0] - 0.5, 0) + max(0.5 - mode_dist_overall[1], 0) * max(mode_dist_balanced[i, 1] - 0.5, 0)
			num_repeat = 1
			if diff > 0:
				num_repeat = np.ceil(1 / (10 * diff))
			print(num_repeat)
			mode_dist_balanced[i, :] = mode_dist_balanced[i, :] * num_repeat
		print('Overall Mode Distribution')
		mode_dist_overall_balanced = mode_dist_balanced.sum(axis=0)
		mode_dist_overall_balanced = mode_dist_overall_balanced/mode_dist_overall_balanced.sum(axis=0)
		print('Major\t\tMinor')
		print(mode_dist_overall_balanced)

	def key_mode_augmentation(self):
		tracks_without_mode_aug = deepcopy(self.tracks)
		num_tracks_without_mode_aug = self.__len__()
		
		self.mode_dist = np.stack([self.mode_dist[:, i] / self.mode_dist.sum(axis=1) for i in range(2)], axis=1)
		mode_dist_overall = self.mode_dist.sum(axis=0)
		mode_dist_overall = mode_dist_overall/mode_dist_overall.sum(axis=0)
		
		mode_dist_balanced = deepcopy(self.mode_dist)
		first_track = tracks_without_mode_aug[0]
		for track in tracks_without_mode_aug:
			diff = max(0.5-mode_dist_overall[0], 0) * (mode_dist_balanced[track - first_track, 0] - 0.5) + max(0.5 - mode_dist_overall[1], 0) * (mode_dist_balanced[track - first_track, 1] - 0.5)
			num_repeat = 1
			if diff > 0:
				num_repeat = round(np.ceil(1 / (10 * diff)))
			for new_track in range(self.tracks[-1] + 1, self.tracks[-1] + num_repeat + 1):
				self.data[new_track] = self.load(track, target_track=new_track)
			self.tracks += list(range(self.tracks[-1] + 1, self.tracks[-1] + num_repeat + 1))



	def init_time_stats(self):
		self.onset_qn_dist = dict()
		self.onset_beat_dist = dict()
		self.duration_qn_dist = dict()
		self.duration_beat_dist = dict()

		self.onset_list = [0.125, 0.25, 0.5, 1, 2]
		for onset in self.onset_list:
			self.onset_qn_dist[onset] = dict()
			self.onset_beat_dist[onset] = dict()
			for track in self.tracks:
				self.onset_qn_dist[onset][track] = 0
				self.onset_beat_dist[onset][track] = 0

		self.duration_list = [0.5 * i for i in range(1, 12)]
		for duration in self.duration_list:
			self.duration_qn_dist[duration] = dict()
			self.duration_beat_dist[duration] = dict()
			for track in self.tracks:
				self.duration_qn_dist[duration][track] = 0
				self.duration_beat_dist[duration][track] = 0

	def print_time_stats(self):
		for onset in self.onset_list:
			onset_qn_dist_cur = np.array(list(self.onset_qn_dist[onset].values()))
			onset_qn_dist_cur = onset_qn_dist_cur / np.sum(onset_qn_dist_cur)

			onset_beat_dist_cur = np.array(list(self.onset_beat_dist[onset].values()))
			onset_beat_dist_cur = onset_beat_dist_cur / np.sum(onset_beat_dist_cur)

			print(f'variance of dist for qn onset {onset} is {np.var(onset_qn_dist_cur):.3}')
			print(f'variance of dist for beat onset {onset} is {np.var(onset_beat_dist_cur):.3}')
		for duration in self.duration_list:
			duration_qn_dist_cur = np.array(list(self.duration_qn_dist[duration].values()))
			duration_qn_dist_cur = onset_qn_dist_cur / np.sum(duration_qn_dist_cur)

			duration_beat_dist_cur = np.array(list(self.duration_beat_dist[duration].values()))
			duration_beat_dist_cur = onset_beat_dist_cur / np.sum(duration_beat_dist_cur)

			print(f'variance of dist for qn duration {duration} is {np.var(duration_qn_dist_cur):.3}')
			print(f'variance of dist for beat duration {duration} is {np.var(duration_beat_dist_cur):.3}')


	def __len__(self):
		"""
		Defines the notion of length for the dataset.

		Returns
		----------
		length : int

		Number of (key-augmented) tracks in the dataset partition
		"""
		length = len(self.tracks)

		return length

	def __getitem__(self, index):
		"""
		Retrieve the randomly sliced data associated with the selected index in Tensor format.

		Parameters
		----------
		index : int

		Index of sampled track

		Returns
		----------
		data : dict

		Dictionary containing the sliced ground-truth data for the sampled track
		"""

		# Get the name of the track
		track_id = self.tracks[index]
		
		# Slice the track's ground-truth
		data = self.get_track_data(track_id)

		# Remove unnecessary and un-batchable entries
		data.pop(tools.KEY_OFFSET, None)
		data.pop(tools.KEY_METER, None)

		return data

	def get_track_data(self, track_id, tonic_shift=None, frame_start=None, snap_to_measure=True):
		"""
		TODO
		"""

		if self.store_data:
			# Copy the track's ground-truth data into a local dictionary
			data = deepcopy(self.data[track_id])
		else:
			# Load the track's ground-truth
			data = self.load(track_id)

		if frame_start is None:
			# If a specific starting frame was not provided, sample one randomly
			frame_start = self.rng.randint(0, data[tools.KEY_PC_ACT].shape[-1] - self.sample_size)
		if snap_to_measure:
			# Compute the amount of frames a single measure spans
			frames_per_measure = tools.FRAMES_PER_QUARTER * data[tools.KEY_METER].get_measure_length()
			# Make sure the sampled frame start corresponds to the start of a measure
			frame_start = round(frame_start / frames_per_measure) * frames_per_measure

		# Determine where the sample ends
		frame_stop = frame_start + self.sample_size

		# Loop through the dictionary keys
		for key in data.keys():
			# Check if the dictionary entry is an array
			if isinstance(data[key], np.ndarray) and len(data[key].shape) > 0:
				# Slice along the final axis
				if key in tools.HARMONY_KEYS:
					data[key] = data[key][..., round(frame_start / tools.HARMONY_UNIT_SPAN) : round(frame_stop / tools.HARMONY_UNIT_SPAN)]
				else:
					data[key] = data[key][..., frame_start : frame_stop]

		return data

	def load(self, track, target_track=None):
		"""
		Get the ground truth for a track. If it has already been saved, load it.

		Parameters
		----------
		track : string
		Name of the track to load

		Returns
		----------
		data : dict
		Dictionary with ground-truth for the track
		"""
		# Load the track data if it exists in memory, otherwise instantiate track data

		# Default data to None (not existing)
		data = None

		# Determine the expected path to the track's data
		gt_path = self.get_gt_path(track)

		# Check if an entry for the data exists
		if self.save_data and os.path.exists(gt_path):
			# Load and unpack the data
			data = dict(np.load(gt_path, allow_pickle=True))
			if target_track is not None and self.save_data:
				target_gt_path = self.get_gt_path(target_track)
				# Save the data as a NumPy zip file
				np.savez_compressed(target_gt_path, **data)


		if data is None:
			# Initialize a new dictionary if there is no saved data
			data = dict()
		else:
			for key in tools.NON_ARRAY_KEYS:
				if key in data.keys() and data[key].dtype == object:
					data[key] = data[key].item()

		# Check if an entry for the data exists
		if not(self.save_data and os.path.exists(gt_path)):
			# Initialize a new dictionary
			data = dict()

			original_track = track
			tonic_shift = 0
			if self.key_tonic_aug:
				original_track = int(np.ceil(track / tools.NUM_PC))
				tonic_shift = track % tools.NUM_PC

			# Obtain the meter information for the track
			meter, qn_per_beat = self.read_meter(original_track)

			# Obtain a list of all notes which occur in the track
			notes = self.read_notes(original_track, tonic_shift, qn_per_beat)

			# Obtain a list of all harmony changes which occur in the track
			harmonies = self.read_harmonies(original_track, track, tonic_shift, qn_per_beat)

			# Determine the first tick of the first note
			tick_first = notes[0].onset
			# Determine the final tick of the last note
			tick_final = notes[-1].get_offset()

			# Determine how many ticks lay in the positive and negative ranges
			num_pos_range_ticks = max(0, tick_final) - max(0, tick_first)
			num_neg_range_ticks = min(0, tick_final) - min(0, tick_first)

			frames_per_measure = meter.get_measure_length() * tools.FRAMES_PER_QUARTER

			# Determine how many frames correspond to time before the measure at zero time
			# Pad the negative frames to a full measure
			num_neg_frames = 0
			if num_neg_range_ticks:
				num_neg_frames = frames_per_measure

			# raw frames
			num_pos_frames = math.ceil(num_pos_range_ticks / tools.TICKS_PER_FRAME)

			# If not, pad the last measure to a full measure
			num_pos_frames = math.ceil(num_pos_frames / frames_per_measure) * frames_per_measure

			# TODO - optionally disable negative frames?

			# Determine the total number of frames based off of both ranges,
			# such that frames onsets line up with the start of each measure
			num_frames = num_neg_frames + num_pos_frames

			pitch_class_activity, bass_pitch_class = self.create_note_tensors(notes, num_frames, num_neg_frames)

			harmony_index_gt, harmony_component_gt = self.create_harmony_tensors(harmonies, num_frames, num_neg_frames)

			# Add all relevant entries to the dictionary
			data.update({
				tools.KEY_TRACK : track,
				tools.KEY_OFFSET : num_neg_frames,

				tools.KEY_PC_ACT : pitch_class_activity,
                tools.KEY_BASS_PC : bass_pitch_class,

                tools.KEY_HARMONY_INDEX_GT : harmony_index_gt, 
                tools.KEY_HARMONY_COMPONENT_GT : harmony_component_gt,

                tools.KEY_METER : meter
                })

			if self.save_data:
				np.savez_compressed(gt_path, **data)

		return data

	def read_notes(self, track, tonic_shift, qn_per_beat):

		# Determine the path to the track's note annotations
		notes_path = self.get_notes_path(track)

		# Load the tabulated note data from the csv file as a NumPy array
		note_entries = pd.read_csv(notes_path, header=None).to_numpy()

		# Initialize a list to hold all Note objects
		notes = list()

		for onset_quarter, midi_pitch, morph_pitch, quarter_duration, staff_num, measure_num in note_entries:
			# Convert the onset and duration to ticks
			# TODO - I think we should floor the onset tick and ceiling the duration
			onset_tick = onset_quarter * tools.TICKS_PER_QUARTER
			tick_duration = quarter_duration * tools.TICKS_PER_QUARTER

			if self.beat_as_unit:
				onset_beat = onset_quarter / qn_per_beat
				beat_duration = quarter_duration / qn_per_beat
				onset_tick = onset_beat * tools.TICKS_PER_BEAT
				tick_duration = beat_duration * tools.TICKS_PER_BEAT
			
			if tick_duration:
				# Add the note entry to the tracked list if duration is non-zero
				
				midi_pitch += tonic_shift
				if midi_pitch > tools.NUM_MIDI:
					midi_pitch -= tools.NUM_PC

				notes.append(tools.Note(round(midi_pitch), round(onset_tick), round(tick_duration)))

		# Make sure the notes are sorted before continuing
		notes = sorted(notes, key=lambda x: x.onset)

		return notes

	def read_harmonies(self, original_track, track, tonic_shift, qn_per_beat):
		# Determine the path to the track's chord annotations
		harmony_path = self.get_harmony_path(original_track)

		# Load the tabulated chord data from the xlsx file as a NumPy array
		harmony_entries = pd.read_excel(harmony_path, header=None).to_numpy()

		# Initialize a list to hold all Harmony objects
		harmonies = list()
		for onset_quarter, offset_quarter, key, degree, quality, inversion, roman_numeral in harmony_entries:
			# Convert the onset and offset to ticks
			onset_tick = onset_quarter * tools.TICKS_PER_QUARTER
			offset_tick = offset_quarter * tools.TICKS_PER_QUARTER

			temp = onset_quarter
			unit_cur = 4
			while temp != 0:
				unit_cur /= 2
				temp = temp % unit_cur
			
			if onset_quarter > 0:
				self.onset_qn_dist[float(unit_cur)][track] += 1
				if float(offset_quarter - onset_quarter) in self.duration_qn_dist.keys():
					self.duration_qn_dist[float(offset_quarter - onset_quarter)][track] += 1
			if self.beat_as_unit:
				onset_beat = onset_quarter / qn_per_beat
				offset_beat = offset_quarter / qn_per_beat
				temp = onset_beat
				unit_cur = 4
				while temp != 0:
					unit_cur /= 2
					temp = temp % unit_cur
				if onset_quarter > 0:
					self.onset_beat_dist[float(unit_cur)][track] += 1
					if float(offset_beat - onset_beat) in self.duration_beat_dist.keys():
						self.duration_beat_dist[float(offset_beat - onset_beat)][track] += 1
				onset_tick = onset_beat * tools.TICKS_PER_BEAT
				offset_tick = offset_beat * tools.TICKS_PER_BEAT

			if len(harmonies):
				# Make sure there is no overlap (due to annotation errors)
				onset_tick_temp = max(onset_tick, harmonies[-1].get_offset())

				if onset_tick_temp >= offset_tick:
					# TODO - smarter way to deal with overlap?
					# TODO - seems to only occur from 316-322 in track 10
					harmonies[-1].duration = onset_tick - harmonies[-1].onset
				else:
					onset_tick = onset_tick_temp

			# Determine the duration of the chord change in ticks
			tick_duration = offset_tick - onset_tick

			# Check the mode based on whether the letter in the key entry is uppercase
			if key.upper() == key:
				mode = "ionian"
			else:
				mode = "aeolian"

			# Convert the key tonic name into the pitch spelling used in our model
			tonic_ps = tools.CLEAN_TONICS[key.upper()]
			key = tools.Key(tonic_ps, mode)
			key.shift_tonic(tonic_shift)

			# I believe the degree annotation of +4/4 is an error
			if degree == '+4/4':
				degree = '-2'
				quality = 'D7'

			# Convert other entries to the representation used in our model
			degree = tools.Degree(str(degree))
			quality = tools.CLEAN_QUALITIES[quality]
			inversion = tools.INVERSIONS[inversion]
			self.mode_dist[track - self.tracks[0], key.mode_index] += tick_duration
			
			if tick_duration:
				# Add the chord change entry to the tracked list if duration is non-zero
				harmonies.append(tools.Harmony(key, degree, quality, inversion, onset_tick, tick_duration))
		# Make sure the harmonies are sorted before continuing
		harmonies = sorted(harmonies, key=lambda x: x.onset)

		return harmonies

	def read_meter(self, track):

		# Determine the paths to the track's beat and downbeat annotations, respectively
		beats_path, downbeats_path = self.get_beats_path(track), self.get_downbeats_path(track)

		# Load the tabulated data from the xlsx files as a NumPy arrays
		beat_entries = pd.read_excel(beats_path, header=None).to_numpy().flatten()
		downbeat_entries = pd.read_excel(downbeats_path, header=None).to_numpy().flatten()

		# Infer the quarter-note values for a beat and a downbeat
		qn_per_beat = np.median(np.diff(beat_entries))
		qn_per_downbeat = np.median(np.diff(downbeat_entries))

		# Compute the metrical components from the inferred beat and downbeat values
		count = qn_per_downbeat / qn_per_beat
		division = 4 * (1 / qn_per_beat)

		# Keep track of the meter information
		meter = tools.Meter(round(count), round(division))

		return meter, qn_per_beat

	def create_note_tensors(self, notes, num_frames, num_neg_frames):

		# Initialize arrays to hold frame-level pitch activity and distribution
		pitch_activity = np.zeros((tools.NUM_PIANO_KEYS, num_frames))

		for note in notes:
			# Determine the frames where the note begins and ends
			frame_onset = round(note.onset / tools.TICKS_PER_FRAME)
			frame_offset = round(note.get_offset() / tools.TICKS_PER_FRAME)

			if num_neg_frames > 0:
				frame_onset += num_neg_frames
				frame_offset += num_neg_frames

			# Account for the pitch activity of the note
			pitch_activity[note.get_key_index(), frame_onset : frame_offset + 1] = 1

		bass_pitch_class = np.zeros((tools.NUM_PC, num_frames))
		for f_i in range(num_frames):
			active_pitches = np.where(pitch_activity[:, f_i] == 1)[0]
			if len(active_pitches) > 0:
				bass_pc = active_pitches[0] % tools.NUM_PC
				bass_pitch_class[bass_pc, f_i] = 1



		# Construct an empty array of activations to complete the first octave
		out_of_bounds_pitches = np.zeros((tools.NUM_PC - tools.NUM_PIANO_KEYS % tools.NUM_PC, num_frames))

		# Append the rest of the first octave to the pitch activations and shift by one to begin at C
		pitch_activity_uncollapsed = np.roll(np.concatenate((out_of_bounds_pitches, pitch_activity)), 1, axis=0)

		# Collapse the pitch activations along the octave dimension to obtain pitch class activations
		pitch_class_activity = np.max(pitch_activity_uncollapsed.reshape(-1, tools.NUM_PC, num_frames), axis=0)

		return pitch_class_activity, bass_pitch_class

	def create_harmony_tensors(self, harmonies, num_frames, num_neg_frames):
		
		# Initialize arrays to hold frame-level harmony label indexes and component indexes
		harmony_index_gt = np.zeros((round(num_frames / tools.HARMONY_UNIT_SPAN),))
		harmony_component_gt = np.zeros((tools.NUM_HARMONY_COMPONENTS[self.harmony_type], round(num_frames / tools.HARMONY_UNIT_SPAN)))

		harmony_index_pre = -1
		for harmony in harmonies:

			# Determine the frames where the harmony begins and ends
			frame_onset = round(harmony.onset / tools.TICKS_PER_FRAME)
			frame_offset = round(harmony.get_offset() / tools.TICKS_PER_FRAME)

			if num_neg_frames > 0:
				frame_onset += num_neg_frames
				frame_offset += num_neg_frames
			
			frame_onset = round(frame_onset / tools.HARMONY_UNIT_SPAN)
			frame_offset = round(frame_offset / tools.HARMONY_UNIT_SPAN)



			# Get the current harmony label index
			if self.harmony_type is tools.HARMONY_TYPE_K:
				harmony_index_cur = harmony.key.get_index()
			elif self.harmony_type is tools.HARMONY_TYPE_RQ:
				harmony_index_cur = harmony.get_rq_index()
			elif self.harmony_type is tools.HARMONY_TYPE_KRQ:
				harmony_index_cur = harmony.get_krq_index()

			# Parse the harmony label index into component indexes
			harmony_component_indexes_cur = tools.Harmony.parse_harmony_index(harmony_index_cur, self.harmony_type)

			# Update label index vector
			harmony_index_gt[frame_onset : frame_offset + 1] = harmony_index_cur
			
			if harmony_index_pre == -1:
				self.harmony_transitions[harmony_index_cur, harmony_index_cur] += frame_offset - frame_onset
			else:
				self.harmony_transitions[harmony_index_pre, harmony_index_cur] += 1
				self.harmony_transitions[harmony_index_cur, harmony_index_cur] += frame_offset - frame_onset
			harmony_index_pre = harmony_index_cur
			
			# Update component index vector
			for i in range(tools.NUM_HARMONY_COMPONENTS[self.harmony_type]):
				harmony_component_gt[i, frame_onset : frame_offset + 1] = harmony_component_indexes_cur[i]

		return harmony_index_gt, harmony_component_gt

	def get_data_stats(self):
		
		key_dist = np.zeros(tools.NUM_KEYS)
		for track in self.available_tracks:
			gt_path = self.get_gt_path(track)
			notes = read_notes








	@abstractmethod
	def get_tracks(self, split):
		"""
		Get the tracks associated with a dataset partition.
		Parameters
		----------
		split : string
		Name of the partition from which to fetch tracks
		"""

		return NotImplementedError



