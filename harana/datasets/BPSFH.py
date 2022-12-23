# Our imports
from .utils import Note, Chord
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
        # TODO - will I actually need all of these?
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
        TODO
        """

        # TODO

        return data

    def get_track_data(self, track_id, sample_start=None, seq_length=None, snap_to_frame=True):
        """
        TODO
        """

        # TODO

        return data

    def load(self, track):
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

        # Determine the expected path to the track data
        gt_path = self.get_gt_path(track)

        # Check if an entry for the data exists
        if self.save_data and os.path.exists(gt_path):
            # Load and unpack the saved data for the track
            data = dict(np.load(gt_path, allow_pickle=True))
        else:
            # Initialize a new dictionary
            data = dict()

            # Obtain a list of all notes which occur in the track
            notes = self.read_notes(track)

            # Obtain a list of all chord changes which occur in the track
            chords = self.read_chords(track)

            # TODO - compute all relevant matrices
            #note_exist_seq = self.tensors_uncollapsed[0][track].cpu().detach().numpy().reshape(-1, 89).T
            #note_dist_seq = self.tensors_uncollapsed[1][track].cpu().detach().numpy().reshape(-1, 89).T
            pc_exist_seq = self.tensors_uncollapsed[2][track].cpu().detach().numpy().reshape(-1, 13).T
            pc_dist_seq = self.tensors_uncollapsed[3][track].cpu().detach().numpy().reshape(-1, 13).T
            #chord_seq = self.tensors_uncollapsed[4][track].cpu().detach().numpy()
            #root_seq = self.tensors_uncollapsed[5][track].cpu().detach().numpy()
            #quality_seq = self.tensors_uncollapsed[6][track].cpu().detach().numpy()
            #key_seq = self.tensors_uncollapsed[7][track].cpu().detach().numpy()
            #rn_seq = self.tensors_uncollapsed[8][track].cpu().detach().numpy()
            #sample_idx_in_song = self.tensors_uncollapsed[10][track].cpu().detach().numpy()
            #qn_offset = self.tensors_uncollapsed[11][track].cpu().detach().numpy()

            # TODO - assumes sorted notes
            # Determine the offset in ticks before zero time
            tick_offset = notes[0].onset
            # Determine the final tick of the last note
            tick_final = notes[-1].get_offset()

            # Compute the total number of ticks/frames for the track
            #num_ticks = notes[-1].get_offset() - tick_offset
            #num_frames = math.ceil(num_ticks / self.ticks_per_frame)
            # Subtract offset so frames begin at the start of a measure
            #num_frames = math.ceil((num_ticks - tick_offset) / self.ticks_per_frame)
            num_frames = math.ceil(tick_final / self.ticks_per_frame)

            pitch_activity = np.zeros((88, num_frames))
            pitch_distr = pitch_activity.copy()

            for note in notes:
                #frame_onset = math.floor((note.onset - tick_offset) / self.ticks_per_frame)
                frame_onset = math.floor(note.onset / self.ticks_per_frame)
                #frame_offset = math.floor((note.get_offset() - tick_offset) / self.ticks_per_frame)
                frame_offset = math.floor(note.get_offset() / self.ticks_per_frame)

                if frame_onset >= 0:
                    pitch_activity[note.get_key_index(), frame_onset : frame_offset + 1] = 1

                    # TODO - this seems like a really weird thing to do
                    for f in range(frame_onset, frame_offset + 1):
                        # TODO - can combine with the frame onset/offset calculation
                        active_ticks = min(note.get_offset() + 1, (f + 1) * self.ticks_per_frame) - \
                                       max(note.onset, f * self.ticks_per_frame)
                        pitch_distr[note.get_key_index(), f : f + 1] += active_ticks

            active_frames = np.sum(pitch_activity, axis=0) > 0

            out_of_bounds_pitches = np.zeros((8, num_frames))
            pitch_activity_uncollapsed = np.roll(np.concatenate((out_of_bounds_pitches, pitch_activity)), 1, axis=0)
            pitch_class_activity = np.max(pitch_activity_uncollapsed.reshape(-1, 12, num_frames), axis=0)

            pitch_distr_uncollapsed = np.roll(np.concatenate((out_of_bounds_pitches, pitch_distr)), 1, axis=0)
            pitch_class_distr = np.sum(pitch_distr_uncollapsed.reshape(-1, 12, num_frames), axis=0)

            pitch_distr[:, active_frames] /= np.sum(pitch_distr[:, active_frames], axis=0)
            pitch_class_distr[:, active_frames] /= np.sum(pitch_class_distr[:, active_frames], axis=0)

            # Add all relevant entries to the dictionary
            data.update({KEY_TRACK : track,
                         'note_exist_seq' : pitch_activity,
                         'note_dist_seq' : pitch_distr,
                         'pc_exist_seq' : pitch_class_activity,
                         'pc_dist_seq' : pitch_class_distr,
                         'chord_seq' : None,
                         'root_seq' : None,
                         'quality_seq' : None,
                         'key_seq' : None,
                         'rn_seq' : None,
                         'sample_idx_in_song' : None,
                         'qn_offset' : None
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
            # TODO - floats or integers?
            onset_tick = onset_quarter * self.ticks_per_quarter
            tick_duration = quarter_duration * self.ticks_per_quarter
            # Add the note entry to the tracked list
            # TODO - best way to handle zero duration?
            notes.append(Note(midi_pitch, onset_tick, tick_duration))

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

            # Use alternate symbols for flat and sharp
            #key.replace('-', 'b').replace('+', '#')

            # Add the chord change entry to the tracked list
            chords.append(Chord(degree, quality, inversion, key, onset_tick, tick_duration, roman_numeral))

        return chords

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
