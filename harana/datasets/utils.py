# Our imports
from .constants import *

# Regular imports
import numpy as np


class Note(object):
    """
    TODO

    TODO - all time here is in ticks
    """

    def __init__(self, midi_pitch, onset, duration):
        """
        TODO
        """

        self.midi_pitch = midi_pitch
        self.onset = onset
        self.duration = duration

    def get_offset(self):
        """
        TODO
        """

        # Compute the last tick where note is active
        offset = self.onset + self.duration - 1

        return offset

    def get_key_index(self):
        """
        TODO
        """

        # Assume first key is A0 (midi pitch 21)
        key_index = self.midi_pitch - MIN_MIDI

        return key_index

    def get_pitch_class_index(self):
        """
        TODO
        """

        # Offset to start at C and obtain remainder
        pitch_class_index = (self.get_key_index() - C_OFFSET) % NUM_PC

        return pitch_class_index


class Degree(object):
    """
    TODO
    """

    def __init__(self, degree):
        """
        TODO
        """

        # Split up the primary and (if available) secondary components of the degree
        components = degree.split('/')
        # Update the notation for flats and sharps
        components = [self.clean_degree(c) for c in components]

        # Extract the primary degree
        self.primary = components[0]
        # Extract the secondary degree if it is available and default to tonic otherwise
        self.secondary = components[-1] if len(components) > 1 else list(FUNCTIONS.keys())[0]

    @staticmethod
    def clean_degree(degree):
        """
        TODO
        """

        # Use alternate symbols for flat and sharp
        degree.replace('-', '♭').replace('+', '♯')

        return degree

    def get_primary_index(self):
        """
        TODO
        """

        # Obtain interval in semitones as the index
        primary_index = FUNCTIONS[self.primary]

        return primary_index

    def get_secondary_index(self):
        """
        TODO
        """

        # Obtain interval in semitones as the index
        secondary_index = FUNCTIONS[self.secondary]

        return secondary_index


class Key(object):
    """
    TODO
    """

    def __init__(self, tonic):
        """
        TODO
        """

        # Determine the root note of the key
        self.tonic = TONICS[tonic.upper()]
        # Determine the mode, indicated by the case of the key
        self.mode = MODES[0] if tonic.isupper() else MODES[1]

    def get_tonic_index(self):
        """
        TODO
        """

        # Obtain an index for the unique pitch spelling of the key
        tonic_index = list(TONICS.values()).index(self.tonic)

        return tonic_index

    def get_mode_index(self):
        """
        TODO
        """

        # Obtain an index for the supported modes
        mode_index = MODES.index(self.mode)

        return mode_index


class Chord(object):
    """
    TODO

    TODO - all time here is in ticks
    """

    def __init__(self, degree, quality, inversion, key, onset, duration):
        """
        TODO
        """

        self.degree = Degree(degree)
        self.quality = quality
        self.inversion = inversion
        self.key = Key(key)
        self.onset = onset
        self.duration = duration

    def get_offset(self):
        """
        TODO
        """

        # Compute the last tick where chord is active
        offset = self.onset + self.duration - 1

        return offset

    def get_quality_index(self):
        """
        TODO
        """

        # Obtain an index for the supported chord qualities
        quality_index = list(CHORD_QUALITIES.values()).index(self.quality)

        return quality_index

    def get_inversion_index(self):
        """
        TODO
        """

        # Obtain an index for the chord inversion
        inversion_index = INVERSIONS.index(self.inversion)

        return inversion_index


class Meter(object):
    """
    TODO
    """

    def __init__(self, count, division):
        """
        TODO
        """

        self.count = count
        self.division = division

    def get_measure_length(self):
        """
        TODO
        """

        # Compute the length of a measure in quarter notes using metric information
        measure_length = round(self.count * (4 / self.division))

        return measure_length
