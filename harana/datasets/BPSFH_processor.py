# Read in the BPSFH dataset and generate a customized pytorch Dataset object for it

# My import
from .base_processor import *

# Regular import
import argparse
import os
import csv
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BPSFHDataset(Dataset):

	def __init__(self, dataset_root_dir, sample_size, frame_type, tpqn, fpqn):
		dataset_name = "BPSFH"

		# The file path of the samples generated from the dataset
		sample_tensors_filepath = os.path.join(os.getcwd(), dataset_root_dir, "BPSFH/sample_tensors")
		# If the samples already exit, read them in
		if os.path.isfile(sample_tensors_filepath):
			self.sample_tensors = torch.load(sample_tensors_filepath)
		# Otherwise, read from the raw dataset
		else:
			# Initialize the data reader for BPSFH dataset
			self.data_reader = BPSFH_DataReader(dataset_root_dir=dataset_root_dir, sample_size=sample_size, dataset_name=dataset_name, tpqn=tpqn, fpqn=fpqn)
			print(f"Data reader initialized for {dataset_name}")

			print("\nReading data...")
			# Read all the data from the dataset
			self.data_reader.read_all()
			print("Done!")

			print("\nConverting data to tensors...")
			# Convert the data to tensors
			self.data_reader.data2sample_tensors(frame_type)
			self.sample_tensors = self.data_reader.sample_tensors
			print("Done!")
			# And save them to the designated filepath
			torch.save(self.sample_tensors, sample_tensors_filepath)
		
		self.num_sample = self.sample_tensors[0].shape[0]




	def __getitem__(self, index):
		sample = {}
		sample["note_exist_seq"] = self.sample_tensors[0][index]
		sample["note_dist_seq"] = self.sample_tensors[1][index]
		sample["pc_exist_seq"] = self.sample_tensors[2][index]
		sample["pc_dist_seq"] = self.sample_tensors[3][index]
		sample["chord_seq"] = self.sample_tensors[4][index]
		sample["root_seq"] = self.sample_tensors[5][index]
		sample["quality_seq"] = self.sample_tensors[6][index]
		sample["key_seq"] = self.sample_tensors[7][index]
		sample["rn_seq"] = self.sample_tensors[8][index]
		sample["song_idx"] = self.sample_tensors[9][index]
		sample["sample_idx_in_song"] = self.sample_tensors[10][index]
		sample["qn_offset"] = self.sample_tensors[11][index]

		return sample

	def __len__(self):
		return self.num_sample

class BPSFH_DataReader(HADataReader):

	# The data reader for the Beethoven Piano Sonata-Functional Harmony dataset

	# A dictionary to convert the pitch spellings used in the BPSFH to those in our vocabulary
	ps_dataset2ours = {
	"C" : "C",
	"C+" : "C#",
	"D-" : "Db",
	"D" : "D",
	"D+" : "D#",
	"E-" : "Eb",
    "E" : "E",
    "F" : "F",
    "F+" : "F#",
    "G-" : "Gb",
    "G" : "G",
    "G+" : "G#",
    "A-" : "Ab",
    "A" : "A",
    "A+" : "A#",
    "B-" : "Bb",
    "B" : "B"
    }

    # A dictionary to convert the chord qualities used in the BPSFH to those in our vocabulary
	chord_quality_dataset2ours = {
	"M" : "maj",
	"m" : "min",
	"a" : "aug",
	"d" : "dim",
	"M7" : "maj7",
	"m7" : "min7",
	"D7" : "dom7",
	"h7" : "hdi7",
	"d7" : "dim7",
	"a6" : "aug6"
	}


	def __init__(self, dataset_root_dir, sample_size, dataset_name, tpqn, fpqn):
		super().__init__(dataset_root_dir, sample_size, dataset_name, tpqn, fpqn)

		# Num of songs in the dataset
		self.num_song = 32

		# Hash tables to find the notes and harmonies of a song according to the song index
		self.notes_all_song = {}
		self.harmonies_all_song = {}

		# The time signature of each song
		self.time_signature_all_song = {}
		
		# The length in quarter notes of the inital imcompelte measure of each song
		self.qn_offset_all_song = {}


	def read_all(self, note=True, harmony=True, meta=True):
		"""
		Read all the data of the dataset

		Parameters
		----------
		note: boolean
			If true, the notes in the dataset will be read in

		harmony: boolean
			If true, the harmonies in the dataset will be read in

		"""

		# The names of the files that store the information of notes and harmonies, respectively
		notes_filename = "notes.csv"
		harmonies_filename = "chords.xlsx"
		meta_filename = "meta"
		print(self.dataset_dir)
		# Walk through all the files and sub-directory in the dataset
		for subdir, dirs, files in os.walk(self.dataset_dir):
			for filename in files:
				if filename.endswith(notes_filename):
					print(f"Reading data : {subdir}")

					# Get the song index
					song_idx = int(subdir.split("/")[-1])
					# Check if each type of information is needed and read them in
					if note:
						self.notes_all_song[song_idx] = self.read_notes(os.path.join(subdir, notes_filename))
					if harmony:
						self.harmonies_all_song[song_idx] = self.read_harmonies(os.path.join(subdir, harmonies_filename))
					if meta:
						meta_info = self.read_meta(os.path.join(subdir, meta_filename))
						self.time_signature_all_song[song_idx] = meta_info["time_signature"]
						self.qn_offset_all_song[song_idx] = meta_info["qn_offset"]
	
	def read_meta(self, meta_dir):
		"""
		Read the meta information of one song in the dataset

		Parameters
		----------
		meta_dir: string
			The directory of the file that stores the meta information of the current song
		"""

		# The dictionary that stores all the meta information
		meta = {}
		# Open the file
		with open(meta_dir, "r") as infile:

			# Iterate through each row, which stores one type of meta information
			for line in infile.readlines():
				# Parse the row and store the information in the dictionary
				meta_name, meta_value = line.rstrip("\n").split(" ")
				meta[meta_name] = meta_value

		return meta

	def read_notes(self, notes_dir):
		"""
		Read the notes of one song in the dataset

		Parameters
		----------
		notes_dir: string
			The directory of the file that stores the note information of the current song
			
		"""
		
		# The list that stores all the notes
		notes = []
		# Open the file
		with open(notes_dir, "r") as infile:

			# Iterate through each row, which stores the information of a single note
			for row in csv.reader(infile):

				# Gather the relevant information
				onset = round(float(row[0]) * self.tpqn)
				midi_num = int(row[1])
				duration = round(float(row[3]) * self.tpqn)

				# Make an object for each note and append the object to the list
				if duration > 0:
					note_cur = core.Note(midi_num, duration, onset, onset + duration - 1)
					notes.append(note_cur)

		return notes

	def read_harmonies(self, chords_dir):
		"""
		Read the harmony information of one song in the dataset

		Parameters
		----------
		chords_dir: string
			The directory of the file that stores the chord information of the current song
			
		"""

		# list to extract the data from the file
		chords_data_all = []

		# Read the excel file as a pandas DataFrame
		xl_file = pd.read_excel(chords_dir)

		# Get the first row 
		first_row = list(xl_file.columns)
		chords_data_all.append(first_row)

		# Get the rest of rows
		for idx, row in xl_file.iterrows():
			chords_data_all.append(list(row))

		# The list that stores all the harmonies
		harmonies = []

		# The list that stors all the keys
		keys = []

		pre_offset = -99
		for row in chords_data_all:

			# In case of overlap of two chord regions (I believe these are annotation errors), use the offset of the first chord as the split point 
			onset = float(row[0])
			offset = float(row[1])
			if onset < pre_offset:
				onset = pre_offset
			pre_offset = offset

			# Convert the time units of chord boundaries from quarter notes to ticks
			onset = onset * self.tpqn
			offset = offset * self.tpqn - 1

			# Get the key
			if row[2].upper() == row[2]:
				key_mode = "maj"
			else:
				key_mode = "min"
			key_tonic_ps = self.ps_dataset2ours[row[2].upper()]
			local_key = key.Key(tonic_ps = key_tonic_ps, mode = key_mode)
			
			# Get the degree
			degree = str(row[3])


			# Get the quality
			quality = self.chord_quality_dataset2ours[str(row[4])]

			# Get the inversion
			inversion = str(row[5])

			# Handle some errors in the dataset
			if degree == "1.1":
				degree = "1"

			if "/" in degree:
				secondary_degree, primary_degree = degree.split("/")
			else:
				primary_degree = degree
				secondary_degree = "1"
			
			if inversion == "0.1":
				inversion = "0"
			
			# Make an object for each chord and append the object to the list
			harmonies.append(harmony.Harmony(local_key = local_key, degree = degree, quality = quality, inversion = inversion, onset = onset, offset = offset))




		return harmonies