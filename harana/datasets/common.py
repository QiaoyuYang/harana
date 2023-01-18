# Common codes to preprocess the datasets

# My import

from .. import tools

# External import
import os
import math
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from abc import abstractmethod
import torch
from torch.utils.data import Dataset


class HarmonyDataset(Dataset):
	
	# A generic class for harmony datasets
	def __init__(self, base_dir, splits, harmony_type, reset_data, store_data, save_data, save_loc, seed):

		# Select a default base directory path if none was provided
		self.base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, self.dataset_name()) if base_dir is None else base_dir

		# Check if the dataset exists (in full) at the specified path
		if not os.path.exists(self.base_dir) or len(os.listdir(self.base_dir)) != len(self.available_tracks()) + 1:
			warnings.warn(f'Dataset was incomplete or could not find at specified '
				f'path \'{self.base_dir}\'. Attempting to download...', category=RuntimeWarning)
			# Download the dataset if it is missing
			self.download(self.base_dir)

		# Choose all available dataset splits if none were provided
		self.splits = self.available_splits() if splits is None else splits

		# Set the storing and saving parameters
		self.store_data = store_data
		self.save_data = save_data
		self.save_loc = tools.DEFAULT_GROUND_TRUTH_DIR if save_loc is None else save_loc

		if os.path.exists(self.get_gt_path()) and reset_data:
			# Remove any saved ground-truth for the dataset
			shutil.rmtree(self.get_gt_path())

		if self.save_data:
			# Make sure the directory for saving/loading ground-truth exists
			os.makedirs(self.get_gt_path(), exist_ok=True)

		# Initialize a random number generator for the dataset
		self.rng = np.random.RandomState(seed)

		self.tracks = list()
		# Aggregate all the track names from the selected splits
		for split in self.splits:
			self.tracks += self.get_tracks(split)

		# Load the ground-truth for each track into RAM
		if self.store_data:
			# Initialize a dictionary to store track data
			self.data = dict()
			for track in tqdm(self.tracks):
				# Load the data for the track
				self.data[track] = self.load(track)

	def __len__(self):
		"""
		Defines the notion of length for the dataset.

		Returns
		----------
		length : int

		Number of tracks in the dataset partition
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

	def get_track_data(self, track_id, frame_start=None, snap_to_measure=True):
		"""
		TODO
		"""

		if self.store_data:
			# Copy the track's ground-truth data into a local dictionary
			data = deepcopy(self.data[track_id])
		else:
			# Load the track's ground-truth
			data = self.load(track_id)

		# TODO - add this back in, otherwise no way to obtain full ground-truth
		"""
		# Check to see if a specific frame length was given
        if frames_per_sample is None:
            if self.frames_per_sample is not None:
                # If not, use the default if available
                frames_per_sample = self.frames_per_sample
            else:
                # Otherwise, assume full track is desired
                return data
		"""

		if frame_start is None:
			# If a specific starting frame was not provided, sample one randomly
			frame_start = self.rng.randint(0, data[tools.KEY_PC_ACT].shape[-1] - tools.FRAMES_PER_SAMPLE)

		if snap_to_measure:
			# TODO - revisit solutions I proposed in Slack to fix this
			# Compute the amount of frames a single measure spans
			frames_per_measure = tools.FRAMES_PER_QUARTER * data[tools.KEY_METER].get_measure_length()
			# Make sure the sampled frame start corresponds to the start of a measure
			frame_start = round(frame_start / frames_per_measure) * frames_per_measure

		# Determine where the sample ends
		frame_stop = frame_start + tools.FRAMES_PER_SAMPLE

		# Loop through the dictionary keys
		for key in data.keys():
			# Check if the dictionary entry is an array
			if isinstance(data[key], np.ndarray):
				# Slice along the final axis
				data[key] = data[key][..., frame_start : frame_stop]

		return data

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

	@abstractmethod
	def load(self, track):
		"""
		Get the ground truth for a track. If it has already been saved, load it.
		If the ground-truth does not exist yet, initialize a new dictionary to hold it.
		Parameters
		----------
		track : string
		Name of the track to load

		Returns
		----------
		data : dict
		Dictionary with ground-truth for the track
		"""

		# Default data to None (not existing)
		data = None

		# Determine the expected path to the track's data
		gt_path = self.get_gt_path(track)

		# Check if an entry for the data exists
		if self.save_data and os.path.exists(gt_path):
			# Load and unpack the data
			data = dict(np.load_dict_npz(gt_path))

		if data is None:
			# Initialize a new dictionary if there is no saved data
			data = dict()
		else:
			# TODO - make sure this works - can't find a reference to NON_ARRAY_KEYS, but
			#        I'm pretty sure it refers to KEY_TRACK, KEY_OFFSET, and KEY_METER
			for key in tools.NON_ARRAY_KEYS:
				if key in data.keys and data[key].dtype == object:
					data[key] = data[key].item()

		return data

	def create_note_tensors(self, notes, num_frames, tick_offset_frame):

		# Initialize arrays to hold frame-level pitch activity and distribution
		pitch_activity = np.zeros((tools.NUM_PIANO_KEYS, num_frames))

		for note in notes:
			# Adjust the onset and offset tick based on the pre-measure ticks
			adjusted_onset_tick = note.onset - tick_offset_frame
			adjusted_offset_tick = note.get_offset() - tick_offset_frame
			# Determine the frames where the note begins and ends
			frame_onset = math.floor(adjusted_onset_tick / tools.TICKS_PER_FRAME)
			frame_offset = math.floor(adjusted_offset_tick / tools.TICKS_PER_FRAME)

			# Account for the pitch activity of the note
			pitch_activity[note.get_key_index(), frame_onset : frame_offset + 1] = 1

		# Construct an empty array of activations to complete the first octave
		out_of_bounds_pitches = np.zeros((tools.NUM_PC - tools.NUM_PIANO_KEYS % tools.NUM_PC, num_frames))

		# Append the rest of the first octave to the pitch activations and shift by one to begin at C
		pitch_activity_uncollapsed = np.roll(np.concatenate((out_of_bounds_pitches, pitch_activity)), 1, axis=0)

		# Collapse the pitch activations along the octave dimension to obtain pitch class activations
		pitch_class_activity = np.max(pitch_activity_uncollapsed.reshape(-1, tools.NUM_PC, num_frames), axis=0)

		return pitch_class_activity

	def create_harmony_tensors(self, harmonies, num_frames, tick_offset_frame):

		# Initialize arrays to hold frame-level harmony label indexes and component indexes
		chord_index_gt = np.zeros((num_frames,))
		rn_index_gt = np.zeros((num_frames,))
		chord_component_gt = np.zeros((tools.NUM_CHORD_COMPONENTS, num_frames))
		rn_component_gt = np.zeros((tools.NUM_RN_COMPONENTS, num_frames))

		for harmony in harmonies:

			# Adjust the onset and offset tick based on the pre-measureticks
			adjusted_onset_tick = harmony.onset - tick_offset_frame
			adjusted_offset_tick = harmony.get_offset() - tick_offset_frame
			# Determine the frames where the note begins and ends
			frame_onset = math.floor(adjusted_onset_tick / tools.TICKS_PER_FRAME)
			frame_offset = math.floor(adjusted_offset_tick / tools.TICKS_PER_FRAME)

			# Get the current harmony label index
			chord_index_cur = harmony.chord.get_index()
			rn_index_cur = harmony.rn.get_index()
			# Parse the harmony label index into component indexes
			chord_component_indexes_cur = tools.Harmony.parse_harmony_index(chord_index_cur, tools.HARMONY_TYPE_CHORD)
			rn_component_indexes_cur = tools.Harmony.parse_harmony_index(rn_index_cur, tools.HARMONY_TYPE_RN)

			# Update label index vector
			chord_index_gt[frame_onset : frame_offset + 1] = chord_index_cur
			rn_index_gt[frame_onset : frame_offset + 1] = rn_index_cur

			# Update component index vector
			for i in range(tools.NUM_CHORD_COMPONENTS):
				chord_component_gt[i, frame_onset : frame_offset + 1] = chord_component_indexes_cur[i]

			for i in range(tools.NUM_RN_COMPONENTS):
				rn_component_gt[i, frame_onset : frame_offset + 1] = rn_component_indexes_cur[i]

		return chord_index_gt, rn_index_gt, chord_component_gt, rn_component_gt

	def get_gt_path(self, track=None):
		"""
        Get the path for the ground-truth directory or a track's ground-truth.

        Parameters
        ----------
        track : string or None
          Append a track to the directory for the track's ground-truth path

        Returns
        ----------
        path : string
          Path to the ground-truth directory or a specific track's ground-truth
        """

		# Get the path to the ground truth directory
		path = os.path.join(self.save_loc, self.dataset_name())

		if track is not None:
			# Add the track name and the .npz extension to the path
			path = os.path.join(path, f'{track}.{tools.NPZ_EXT}')

		return path

	@staticmethod
	def available_splits():
		"""
        Obtain a list of possible splits.
        """

		return NotImplementedError

	@classmethod
	def dataset_name(cls):
		"""
		Retrieve an appropriate tag, the class name, for the dataset.

		Returns
		----------
		tag : str
		  Name of the child class calling the function
		"""

		tag = cls.__name__

		return tag

	@staticmethod
	@abstractmethod
	def download(save_dir):
		"""
		Download the dataset to disk.

		Parameters
		----------
		save_dir : string
		  Directory under which to save the dataset
		"""

		return NotImplementedError
