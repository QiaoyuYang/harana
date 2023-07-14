# Our imports
from .common import HaranaDataset
from .. import tools

# Regular imports
import pandas as pd
import numpy as np
import math
import os
import shutil


class BPSFH(HaranaDataset):
    """
    TODO
    """

    def __init__(self, base_dir=None, splits=None, sample_size=None, harmony_type=None, reset_data=False,
                       store_data=False, save_data=False, save_loc=None, key_tonic_aug=False, key_mode_aug=False, beat_as_unit=False, validation=False, seed=0):
        """
        TODO
        """
        super().__init__(base_dir, splits, sample_size, harmony_type, reset_data, store_data, save_data, save_loc, key_tonic_aug, key_mode_aug, beat_as_unit, validation, seed)

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
        if self.key_tonic_aug:
            split_span = 8 * tools.NUM_TONICS
        else:
            split_span = 8
        
        # Determine where the split starts within the sorted tracks
        split_start = int(split) * split_span
        # Slice the appropriate tracks
        tracks = list(range(split_start + 1, split_start + split_span + 1))

        return tracks

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
        if self.key_tonic_aug and self.key_mode_aug:
            key_aug_type = 'key_tonic_mode_aug'
        if self.key_tonic_aug and (not self.key_mode_aug):
            key_aug_type = 'key_tonic_aug'
        if (not self.key_tonic_aug) and self.key_mode_aug:
            key_aug_type = 'key_mode_aug'
        if (not self.key_tonic_aug) and (not self.key_mode_aug):
            key_aug_type = 'no_key_aug'
        path = os.path.join(self.save_loc, self.dataset_name(), key_aug_type, 'validation' if self.validation else 'train')

        if track is not None:
            # Add the track name and the .npz extension to the path
            path = os.path.join(path, f'{track}.{tools.NPZ_EXT}')

        return path

    def get_scores_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's note annotations
        notes_path = os.path.join(self.base_dir, f'{track}', f'scores.{tools.CSV_EXT}')

        return notes_path

    def get_harmony_path(self, track):
        """
        TODO
        """

        # Construct the path to a track's chord annotations
        chords_path = os.path.join(self.base_dir, f'{track}', f'chords.{tools.XLSX_EXT}')

        return chords_path

    @staticmethod
    def available_tracks():
        """
        TODO
        """

        # Track names are integers ranging from 1 to 32
        available_tracks = [track_id for track_id in (np.arange(32) + 1)]

        return available_tracks

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. The splits are by piece indexes.
        Each split includes four pieces consecutive in index.

        Returns
        ----------
        splits : list of strings
          Player codes listed at beginning of file names
        """

        splits = ['00', '01', '02', '03']

        return splits

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
