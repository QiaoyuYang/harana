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

	def __init__(self, dataset_root_dir, frame_type, fpqn):
		dataset_name = "BPSFH"

		# The file path of the samples generated from the dataset
		sample_tensors_filepath = os.path.join(os.getcwd(), dataset_root_dir, "BPSFH.sample_tensors")

		# If the samples already exit, read them in
		if os.path.isfile(sample_tensors_filepath):
			self.sample_tensors = torch.load(sample_tensors_filepath)
		# Otherwise, read from the raw dataset
		else:
			# Initialize the data reader for BPSFH dataset
			self.data_reader = BPSFH_DataReader(dataset_root_dir = dataset_root_dir, dataset_name = dataset_name, fpqn = fpqn)
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


	def __getitem__(self, index):
		data = {}
		data["notes_piano_roll"] = self.sample_tensors[0][index]
		data["notes_pc_roll_existence"] = self.sample_tensors[1][index]
		data["notes_pc_roll_accumulation"] = self.sample_tensors[2][index]
		data["chord_sequence"] = self.sample_tensors[3][index]

		return data

class BPSFH_DataReader(HADataReader):

	# The data reader for the Beethoven Piano Sonata-Functional Harmony dataset

	# A dictionary to convert the pitch names used in the BPSFH to those in our vocabulary
	pn_dataset2ours_dict = {
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
	chord_quality_dataset2ours_dict = {
	"M" : "M",
	"m" : "m",
	"a" : "+",
	"d" : "-",
	"M7" : "Maj7",
	"m7" : "min7",
	"D7" : "Dom7",
	"d7" : "dim7",
	"h7" : "hdi7"
	}


	def __init__(self, dataset_root_dir, dataset_name, fpqn):
		super().__init__(dataset_root_dir, dataset_name, fpqn)

		# Num of songs in the dataset
		self.num_song = 32

		# Hash tables to find the notes and chords of a song according to the song index
		self.notes_all_song = {}
		self.chords_all_song = {}


	def read_all(self, note=True, chord=True):
		"""
		Read all the data of the dataset

		Parameters
		----------
		note: boolean
			If true, the notes in the dataset will be read in

		chord: boolean
			If true, the chords in the dataset will be read in

		"""

		# The names of the files that store the information of notes and chords, respectively
		notes_filename = "notes.csv"
		chords_filename = "chords.xlsx"
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
					if chord:
						self.chords_all_song[song_idx] = self.read_chords(os.path.join(subdir, chords_filename))

	
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
				onset = round(float(row[0]) * self.fpqn)
				midi_num = int(row[1])
				duration = round(float(row[3]) * self.fpqn)

				# Make an object for each note and append the object to the list
				note_cur = core.Note(midi_num, duration, onset, onset + duration)
				notes.append(note_cur)

		return notes

	def read_chords(self, chords_dir):
		"""
		Read the chords of one song in the dataset

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

		# The list that stores all the notes
		chords = []

		pre_offset = -99
		for row in chords_data_all:

			# In case of overlap of two chord regions (I believe these are annotation errors), use the offset of the first chord as the split point 
			onset = float(row[0])
			offset = float(row[1])
			if onset < pre_offset:
				onset = pre_offset
			pre_offset = offset

			# Convert the time units of chord boundaries from quarter notes to ticks
			onset = round(onset * self.fpqn)
			offset = round(offset * self.fpqn)

			# Get the pitch class of the key
			key_pc = core.pn2pc_dict[self.pn_dataset2ours_dict[row[2].upper()]]
			
			# Get the degree
			degree = str(row[3])

			# Get the inversion
			inversion = str(row[5])

			# Handle some errors in the dataset
			if degree == "1.1":
				degree = "1"

			if inversion == "0.1":
				inversion = "0"

			# Split the primary degree and the secondary degree
			# Get the pitch class of root from the key and the degree
			if "/" in degree:
				secondary_degree, primary_degree = degree.split("/")
				root_pc = (key_pc + chord.degree2pc_add_dict[primary_degree] + chord.degree2pc_add_dict[secondary_degree])%12
			else:
				root_pc = (key_pc + chord.degree2pc_add_dict[degree])%12

			# Get the quality
			quality = str(row[4])

			# Get the specific type of augmented sixth chord if there is one because they have different chordal notes
			# The type is specified by the roman numeral entry
			if quality == "a6":
				quality = row[6]
				if "/" in quality:
					quality = row[6].split("/")[0]
			else:
				quality = self.chord_quality_dataset2ours_dict[quality]

			# Get the pitch classes of the chordal notes of the chord
			chordal_pc = chord.get_chordal_pc(root_pc, quality)

			# Get the pitch class of the bass note depending on the inversion
			if inversion == "0":
				bass_pc = chordal_pc[0]
			if inversion == "1":
				bass_pc = chordal_pc[1]
			if inversion == "2":
				bass_pc = chordal_pc[2]
			if inversion == "3":
				bass_pc = chordal_pc[3]


			# Make an object for each chord and append the object to the list
			chords.append(chord.Chord(root_pc, quality, bass_pc, onset, offset))

		return chords