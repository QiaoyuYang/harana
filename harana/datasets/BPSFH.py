# Our imports
from .utils import Note, Chord, Meter
from .constants import *

# Regular imports
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm

import pandas as pd
import numpy as np
import warnings
import shutil
import torch
import math
import os


class BPSFH(Dataset):
    """
    TODO
    """
    # TODO - remove after verifying same output
    tensors_collapsed = torch.load('harana/datasets/sample_tensors_collapsed')
    tensors_uncollapsed = torch.load('harana/datasets/sample_tensors_uncollapsed')

    def __init__(self, base_dir=None, tracks=None, ticks_per_quarter=24,
                       frames_per_quarter=4, frames_per_sample=8, reset_data=False,
                       store_data=False, save_data=False, save_loc=None, seed=0):
        """
        TODO
        """

        # Select a default base directory path if none was provided
        self.base_dir = os.path.join(DEFAULT_DATASETS_DIR, self.dataset_name()) if base_dir is None else base_dir

        # Check if the dataset exists (in full) at the specified path
        if not os.path.exists(self.base_dir) or len(os.listdir(self.base_dir)) != len(self.available_tracks()) + 1:
            warnings.warn(f'Dataset was incomplete or could not find at specified '
                          f'path \'{self.base_dir}\'. Attempting to download...', category=RuntimeWarning)
            # Download the dataset if it is missing
            self.download(self.base_dir)

        # Set the parameters related to ticks and frames
        # TODO - do/can all of these actually vary for this dataset? - if not make constants...
        self.ticks_per_quarter = ticks_per_quarter
        self.frames_per_quarter = frames_per_quarter
        self.frames_per_sample = frames_per_sample
        self.ticks_per_frame = ticks_per_quarter / self.frames_per_quarter

        # Set the storing and saving parameters
        self.store_data = store_data
        self.save_data = save_data
        self.save_loc = DEFAULT_GROUND_TRUTH_DIR if save_loc is None else save_loc

        if os.path.exists(self.get_gt_path()) and reset_data:
            # Remove any saved ground-truth for the dataset
            shutil.rmtree(self.get_gt_path())

        if self.save_data:
            # Make sure the directory for saving/loading ground-truth exists
            os.makedirs(self.get_gt_path(), exist_ok=True)

        # Initialize a random number generator for the dataset
        self.rng = np.random.RandomState(seed)

        # Choose all available tracks if none were provided
        self.tracks = self.available_tracks() if tracks is None else tracks

        # Verify the validity of all chosen tracks
        assert np.all([t in self.available_tracks() for t in self.tracks])

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
        data.pop(KEY_OFFSET, None)
        data.pop(KEY_METER, None)

        return data

    def get_track_data(self, track_id, frame_start=None, frames_per_sample=None, snap_to_measure=True):
        """
        TODO
        """

        if self.store_data:
            # Copy the track's ground-truth data into a local dictionary
            data = deepcopy(self.data[track_id])
        else:
            # Load the track's ground-truth
            data = self.load(track_id)

        # Check to see if a specific frame length was given
        if frames_per_sample is None:
            if self.frames_per_sample is not None:
                # If not, use the default if available
                frames_per_sample = self.frames_per_sample
            else:
                # Otherwise, assume full track is desired
                return data

        if frame_start is None:
            # If a specific starting frame was not provided, sample one randomly
            frame_start = self.rng.randint(0, data[KEY_PC_ACT].shape[-1] - frames_per_sample)

        if snap_to_measure:
            # Compute the amount of frames a single measure spans
            measure_length = self.frames_per_quarter * data[KEY_METER].get_measure_length()
            # Make sure the sampled frame start corresponds to the start of a measure
            frame_start = round(frame_start / measure_length) * measure_length

        # Determine where the sample ends
        frame_stop = frame_start + frames_per_sample

        # Loop through the dictionary keys
        for key in data.keys():
            # Check if the dictionary entry is an array
            if isinstance(data[key], np.ndarray):
                # Slice along the final axis
                data[key] = data[key][..., frame_start : frame_stop]

        return data

    def load(self, track):
        """
        Get the ground truth for a track. If it has already been saved, load it.

        TODO - can potentially break this function up

        Parameters
        ----------
        track : string
          Name of the track to load

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """

        # Determine the expected path to the track data
        gt_path = self.get_gt_path(track)

        # Check if an entry for the data exists
        if self.save_data and os.path.exists(gt_path):
            # Load and unpack the saved data for the track
            data = dict(np.load(gt_path, allow_pickle=True))
            # Unpack the track ID, frame offset, and meter from their respective arrays
            data[KEY_TRACK] = data[KEY_TRACK].item()
            data[KEY_OFFSET] = data[KEY_OFFSET].item()
            data[KEY_METER] = data[KEY_METER].item()
        else:
            # Initialize a new dictionary
            data = dict()

            # Obtain a list of all notes which occur in the track
            notes = self.read_notes(track)

            # Make sure the notes are sorted before continuing
            notes = sorted(notes, key=lambda x: x.onset)

            # Determine the offset in ticks before zero time
            tick_offset = notes[0].onset
            # Determine the final tick of the last note
            tick_final = notes[-1].get_offset()

            # Determine how many ticks lay in the positive and negative ranges
            num_pos_range_ticks = max(0, tick_final) - max(0, tick_offset)
            num_neg_range_ticks = min(0, tick_final) - min(0, tick_offset)

            # Determine how many frames correspond to time before the measure at zero time
            num_neg_frames = math.ceil(num_neg_range_ticks / self.ticks_per_frame)

            # TODO - optionally disable negative frames?

            # Compute the global tick offset needed to start with a full frame
            tick_offset_frame = -(num_neg_frames * self.ticks_per_frame)

            # Determine the total number of frames based off of both ranges,
            # such that frames onsets line up with the start of each measure
            num_frames = num_neg_frames + math.ceil(num_pos_range_ticks / self.ticks_per_frame)

            # Initialize arrays to hold frame-level pitch activity and distribution
            pitch_activity = np.zeros((NUM_KEYS, num_frames))
            pitch_distr = pitch_activity.copy()

            for note in notes:
                # Adjust the onset and offset tick based on the pre-measureticks
                adjusted_onset_tick = note.onset - tick_offset_frame
                adjusted_offset_tick = note.get_offset() - tick_offset_frame
                # Determine the frames where the note begins and ends
                frame_onset = math.floor(adjusted_onset_tick / self.ticks_per_frame)
                frame_offset = math.floor(adjusted_offset_tick / self.ticks_per_frame)

                # Account for the pitch activity of the note
                pitch_activity[note.get_key_index(), frame_onset : frame_offset + 1] = 1
                # Loop through each frame where the note is active
                for f in range(frame_onset, frame_offset + 1):
                    # Determine the amount of ticks during the frame where the note is active
                    active_ticks = min(adjusted_offset_tick + 1, (f + 1) * self.ticks_per_frame) - \
                                   max(adjusted_onset_tick, f * self.ticks_per_frame)
                    # Add a score (number of active ticks) for the note at this frame
                    # TODO - adding will artificially increase strength of duplicated notes
                    pitch_distr[note.get_key_index(), f : f + 1] += active_ticks

            # Determine which frames contain pitch activity
            active_frames = np.sum(pitch_activity, axis=0) > 0

            # Construct an empty array of activations to complete the first octave
            out_of_bounds_pitches = np.zeros((NUM_PC - NUM_KEYS % NUM_PC, num_frames))

            # Append the rest of the first octave to the pitch activations and shift by one to begin at C
            pitch_activity_uncollapsed = np.roll(np.concatenate((out_of_bounds_pitches, pitch_activity)), 1, axis=0)
            # Collapse the pitch activations along the octave dimension to obtain pitch class activations
            pitch_class_activity = np.max(pitch_activity_uncollapsed.reshape(-1, NUM_PC, num_frames), axis=0)

            # Append the rest of the first octave to the pitch distributions and shift by one to begin at C
            pitch_distr_uncollapsed = np.roll(np.concatenate((out_of_bounds_pitches, pitch_distr)), 1, axis=0)
            # Collapse the pitch distributions along the octave dimension to obtain pitch class distributions
            pitch_class_distr = np.sum(pitch_distr_uncollapsed.reshape(-1, NUM_PC, num_frames), axis=0)

            # Normalize the pitch distributions to obtain probability-like values
            pitch_distr[:, active_frames] /= np.sum(pitch_distr[:, active_frames], axis=0)
            # Normalize the pitch class distributions to obtain probability-like values
            pitch_class_distr[:, active_frames] /= np.sum(pitch_class_distr[:, active_frames], axis=0)

            # TODO - verify equivalence of values across all tracks, then remove the following 9 lines
            num_pos_frames = math.ceil(num_pos_range_ticks / self.ticks_per_frame)
            note_exist_seq = self.tensors_uncollapsed[0][track].cpu().detach().numpy().reshape(-1, 89).T
            note_dist_seq = self.tensors_uncollapsed[1][track].cpu().detach().numpy().reshape(-1, 89).T
            pc_exist_seq = self.tensors_uncollapsed[2][track].cpu().detach().numpy().reshape(-1, 13).T
            pc_dist_seq = self.tensors_uncollapsed[3][track].cpu().detach().numpy().reshape(-1, 13).T
            print()
            print(np.allclose(pitch_activity[..., num_neg_frames : num_frames - 8], note_exist_seq[:-1, : num_pos_frames - 8]))
            print(np.allclose(pitch_distr[..., num_neg_frames : num_frames - 8], note_dist_seq[:-1, : num_pos_frames - 8]))
            print(np.allclose(pitch_class_activity[..., num_neg_frames : num_frames - 8], pc_exist_seq[:-1, : num_pos_frames - 8]))
            print(np.allclose(pitch_class_distr[..., num_neg_frames : num_frames - 8], pc_dist_seq[:-1, : num_pos_frames - 8]))

            # Obtain a list of all chord changes which occur in the track
            chords = self.read_chords(track)

            # Obtain the meter information for the track
            meter = self.read_meter(track)

            # Add all relevant entries to the dictionary
            data.update({
                KEY_TRACK : track,
                KEY_PC_ACT : pitch_class_activity,
                KEY_PC_DST : pitch_class_distr,
                KEY_OFFSET : num_neg_frames,
                # TODO - chords data
                KEY_METER : meter
            })

            if self.save_data:
                # Save the data as a NumPy zip file
                np.savez_compressed(gt_path, **data)

        return data

    def read_notes(self, track):
        """
        TODO
        """

        # Determine the path to the track's note annotations
        notes_path = self.get_notes_path(track)

        # Load the tabulated note data from the csv file as a NumPy array
        note_entries = pd.read_csv(notes_path, header=None).to_numpy()

        # Initialize a list to hold all Note objects
        notes = list()

        for onset_quarter, midi_pitch, morph_pitch, \
                quarter_duration, staff_num, measure_num in note_entries:
            # Convert the onset and duration to ticks
            # TODO - I think we should floor the onset tick and ceiling the duration
            onset_tick = onset_quarter * self.ticks_per_quarter
            tick_duration = quarter_duration * self.ticks_per_quarter

            if tick_duration:
                # Add the note entry to the tracked list if duration is non-zero
                notes.append(Note(round(midi_pitch), round(onset_tick), round(tick_duration)))

        return notes

    def read_chords(self, track):
        """
        TODO
        """

        # Determine the path to the track's chord annotations
        chords_path = self.get_chords_path(track)

        # Load the tabulated chord data from the xlsx file as a NumPy array
        chord_entries = pd.read_excel(chords_path, header=None).to_numpy()

        # Initialize a list to hold all Chord objects
        chords = list()

        for onset_quarter, offset_quarter, key, degree, \
                quality, inversion, roman_numeral in chord_entries:
            # Convert the onset and offset to ticks
            onset_tick = onset_quarter * self.ticks_per_quarter
            offset_tick = offset_quarter * self.ticks_per_quarter

            if len(chords):
                # Make sure there is no overlap (due to annotation errors)
                onset_tick = max(onset_tick, chords[-1].get_offset())

                if onset_tick >= offset_tick:
                    # TODO - smarter way to deal with overlap?
                    # TODO - seems to only occur from 316-322 in track 10
                    continue

            # Determine the duration of the chord change in ticks
            tick_duration = offset_tick - onset_tick

            # Convert fields to appropriate string representations
            degree = str(degree)
            quality = CHORD_QUALITIES[quality]
            inversion = INVERSIONS[inversion]

            if tick_duration:
                # Add the chord change entry to the tracked list if duration is non-zero
                chords.append(Chord(degree, quality, inversion, key, onset_tick, tick_duration))

        return chords

    def read_meter(self, track):
        """
        TODO
        """

        # Determine the paths to the track's beat and downbeat annotations, respectively
        beats_path, downbeats_path = self.get_beats_path(track), self.get_downbeats_path(track)

        # Load the tabulated data from the xlsx files as a NumPy arrays
        beat_entries = pd.read_excel(beats_path, header=None).to_numpy().flatten()
        downbeat_entries = pd.read_excel(downbeats_path, header=None).to_numpy().flatten()

        # Infer the quarter-note values for a beat and a downbeat
        beat_quarter = np.median(np.diff(beat_entries))
        downbeat_quarter = np.median(np.diff(downbeat_entries))

        # Compute the metrical components from the inferred beat and downbeat values
        count = downbeat_quarter / beat_quarter
        division = 4 * (1 / beat_quarter)

        # Keep track of the meter information
        meter = Meter(round(count), round(division))

        return meter

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
            path = os.path.join(path, f'{track}.{NPZ_EXT}')

        return path

    def get_notes_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's note annotations
        notes_path = os.path.join(self.base_dir, f'{track}', f'notes.{CSV_EXT}')

        return notes_path

    def get_chords_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's chord annotations
        chords_path = os.path.join(self.base_dir, f'{track}', f'chords.{XLSX_EXT}')

        return chords_path

    def get_beats_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's beat annotations
        beats_path = os.path.join(self.base_dir, f'{track}', f'beats.{XLSX_EXT}')

        return beats_path

    def get_downbeats_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's downbeat annotations
        downbeats_path = os.path.join(self.base_dir, f'{track}', f'dBeats.{XLSX_EXT}')

        if not os.path.exists(downbeats_path):
            # There is no capital B in the file name for some tracks...
            downbeats_path = os.path.join(os.path.dirname(downbeats_path),
                                          os.path.basename(downbeats_path).lower())

        return downbeats_path

    @staticmethod
    def available_tracks():
        """
        TODO
        """

        # Track names are integers ranging from 1 to 32
        available_tracks = np.arange(32) + 1

        return available_tracks

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
    def download(save_dir):
        """
        Download the dataset to disk.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the dataset
        """

        if os.path.isdir(save_dir):
            # Remove preexisting directory
            shutil.rmtree(save_dir)

        # Create the base directory
        os.makedirs(save_dir)

        # Download the dataset to the specified path using Git clone
        os.system(f'git clone https://github.com/Tsung-Ping/functional-harmony {save_dir}/temp')
        # Move the dataset contents to the top-level directory
        os.system(f'mv {save_dir}/temp/BPS_FH_Dataset/* {save_dir}')
        # Remove the extraneous contents of the GitHub repository
        os.system(f'rm -rf {save_dir}/temp')
