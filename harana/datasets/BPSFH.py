# Our imports
from .common import HarmonyDataset
from .. import tools

# Regular imports
import pandas as pd
import numpy as np
import shutil
import math
import os


class BPSFH(HarmonyDataset):
    """
    TODO
    """

    def __init__(self, base_dir=None, splits=None, reset_data=False,
                 store_data=False, save_data=False, save_loc=None, seed=0):
        """
        TODO
        """

        super().__init__(base_dir, splits, reset_data, store_data, save_data, save_loc, seed)

    def get_tracks(self, split):
        """
        Get the tracks associated with a dataset partition.

        Parameters
        ----------
        split : string
          Name of the partition from which to fetch tracks

        Returns
        ----------
        tracks : list of strings
          Names of tracks within the given partition
        """

        # Get all the available tracks
        tracks = self.available_tracks()

        # Determine where the split starts within the sorted tracks
        split_start = int(split) * 8
        
        # Slice the appropriate tracks in groups of 8
        tracks = tracks[split_start : split_start + 8]

        return tracks
    
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

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = super().load(track)

        # If the track data is being instantiated, it will not have the track key
        if not tools.KEY_TRACK in data:
            # Obtain a list of all notes which occur in the track
            notes = self.read_notes(track)

            # Obtain a list of all harmony changes which occur in the track
            harmonies = self.read_harmonies(track)

            # Obtain the meter information for the track
            meter = self.read_meter(track)

            # Determine the offset in ticks before zero time
            tick_offset = notes[0].onset
            # Determine the final tick of the last note
            tick_final = notes[-1].get_offset()

            # Determine how many ticks lay in the positive and negative ranges
            num_pos_range_ticks = max(0, tick_final) - max(0, tick_offset)
            num_neg_range_ticks = min(0, tick_final) - min(0, tick_offset)

            # Determine how many frames correspond to time before/after the measure at zero time
            num_neg_frames = math.ceil(num_neg_range_ticks / tools.TICKS_PER_FRAME)
            num_pos_frames = math.ceil(num_pos_range_ticks / tools.TICKS_PER_FRAME)

            # Compute the amount of frames a single measure spans
            frames_per_measure = tools.FRAMES_PER_QUARTER * meter.get_measure_length()

            # Pad frames on each side of zero time to start and end on measure divisions
            # TODO - do we want to pad for full measures here or under snap_to_measure=True?
            #        what do the chord labels look like for these padded frames?
            #        is it OK to always do this?
            num_neg_frames = math.ceil(num_neg_frames / frames_per_measure) * frames_per_measure
            num_pos_frames = math.ceil(num_pos_frames / frames_per_measure) * frames_per_measure

            # Compute the global tick offset needed to start with a full frame
            tick_offset_frame = -(num_neg_frames * tools.TICKS_PER_FRAME)

            # Determine the total number of frames based off of both ranges
            num_frames = num_neg_frames + num_pos_frames

            pitch_class_activity = self.create_note_tensors(notes, num_frames, tick_offset_frame)

            chord_index_gt, rn_index_gt, chord_component_gt, rn_component_gt = self.create_harmony_tensors(harmonies, num_frames, tick_offset_frame)

            # Add all relevant entries to the dictionary
            data.update({
                tools.KEY_TRACK : track,
                
                tools.KEY_PC_ACT : pitch_class_activity,
                
                tools.KEY_CHORD_INDEX_GT : chord_index_gt, 
                tools.KEY_RN_INDEX_GT : rn_index_gt,
                tools.KEY_CHORD_COMPONENT_GT : chord_component_gt,
                tools.KEY_RN_COMPONENT_GT : rn_component_gt,

                tools.KEY_METER : meter
            })

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_path(track)

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
            onset_tick = onset_quarter * tools.TICKS_PER_QUARTER
            tick_duration = quarter_duration * tools.TICKS_PER_QUARTER

            if tick_duration:
                # Add the note entry to the tracked list if duration is non-zero
                notes.append(tools.Note(round(midi_pitch), round(onset_tick), round(tick_duration)))

        # Make sure the notes are sorted
        notes = sorted(notes, key=lambda x: x.onset)

        return notes

    def read_harmonies(self, track):
        """
        TODO
        """

        # Determine the path to the track's chord annotations
        harmony_path = self.get_harmony_path(track)

        # Load the tabulated chord data from the xlsx file as a NumPy array
        harmony_entries = pd.read_excel(harmony_path, header=None).to_numpy()

        # Initialize a list to hold all Harmony objects
        harmonies = list()

        for onset_quarter, offset_quarter, key, degree, \
                quality, inversion, roman_numeral in harmony_entries:
            # Convert the onset and offset to ticks
            onset_tick = onset_quarter * tools.TICKS_PER_QUARTER
            offset_tick = offset_quarter * tools.TICKS_PER_QUARTER

            if len(harmonies):
                # Make sure there is no overlap (due to annotation errors)
                onset_tick_temp = max(onset_tick, harmonies[-1].get_offset())

                # TODO - additional logic has been added here
                if onset_tick_temp >= offset_tick:
                    # TODO - smarter way to deal with overlap?
                    # TODO - seems to only occur from 316-322 in track 10
                    harmonies[-1].duration = onset_tick - harmonies[-1].onset
                else:
                    onset_tick = onset_tick_temp

            # Determine the duration of the chord change in ticks
            tick_duration = offset_tick - onset_tick

            # TODO - below code has changed slightly

            # Check the mode based on whether the letter in the key entry is uppercase
            if key.upper() == key:
                mode = "ionian"
            else:
                mode = "aeolian"

            # Convert the key tonic name into the pitch spelling used in our model
            tonic_ps = tools.CLEAN_TONICS[key.upper()]
            key = tools.Key(tonic_ps, mode)

            # TODO - I believe the degree annotation of +4/4 is an error
            if degree == '+4/4':
                degree = '-2'
                quality = 'D7'
            
            # Convert other entries to the representation used in our model
            degree = tools.Degree(str(degree))
            quality = tools.CLEAN_QUALITIES[quality]
            inversion = tools.INVERSIONS[inversion]

            if tick_duration:
                # Add the chord change entry to the tracked list if duration is non-zero
                harmonies.append(tools.Harmony(key, degree, quality, inversion, onset_tick, tick_duration))

        # Make sure the harmonies are sorted before continuing
        harmonies = sorted(harmonies, key=lambda x: x.onset)

        return harmonies

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
        meter = tools.Meter(round(count), round(division))

        return meter

    def get_notes_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's note annotations
        notes_path = os.path.join(self.base_dir, f'{track}', f'notes.{tools.CSV_EXT}')

        return notes_path

    def get_harmony_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's chord annotations
        chords_path = os.path.join(self.base_dir, f'{track}', f'chords.{tools.XLSX_EXT}')

        return chords_path

    def get_beats_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's beat annotations
        beats_path = os.path.join(self.base_dir, f'{track}', f'beats.{tools.XLSX_EXT}')

        return beats_path

    def get_downbeats_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's downbeat annotations
        downbeats_path = os.path.join(self.base_dir, f'{track}', f'dBeats.{tools.XLSX_EXT}')

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
        available_tracks = list(np.arange(32) + 1)

        return available_tracks

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits by groups of consecutive piece indices.

        Returns
        ----------
        splits : list of strings
          Groups of consecutive indices
        """

        splits = ['00', '01', '02', '03']

        return splits

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
