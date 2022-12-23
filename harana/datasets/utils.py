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
        # TODO - don't need cast if we always add integers in the first place
        key_index = int(self.midi_pitch - 21)

        return key_index

    def get_pitch_class_index(self):
        """
        TODO
        """

        # Offset to start at C and obtain remainder
        pitch_class_index = (self.get_key_index() - 3) % 12

        return pitch_class_index


class Degree(object):
    """
    TODO
    """

    def __init__(self, degree):
        """
        TODO
        """

        components = degree.split('/')

        self.primary = self.clean_degree(components[0])
        self.secondary = self.clean_degree(components[-1]) if len(components) > 1 else list(FUNCTIONS.keys())[0]

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

        primary_index = FUNCTIONS[self.primary]

        return primary_index

    def get_secondary_index(self):
        """
        TODO
        """

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

        self.tonic = TONICS[tonic.upper()]
        self.mode = MODES[0] if tonic.isupper() else MODES[1]

    def get_tonic_index(self):
        """
        TODO
        """

        tonic_index = list(TONICS.values()).index(self.tonic)

        return tonic_index

    def get_mode_index(self):
        """
        TODO
        """

        mode_index = MODES.index(self.mode)

        return mode_index


class Chord(object):
    """
    TODO

    TODO - all time here is in ticks
    """

    def __init__(self, degree, quality, inversion, key, onset, duration, roman_numeral):
        """
        TODO
        """

        self.degree = Degree(degree)
        self.quality = quality
        self.inversion = inversion
        self.key = Key(key)
        self.onset = onset
        self.duration = duration
        self.roman_numeral = roman_numeral

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

        quality_index = list(CHORD_QUALITIES.values()).index(self.quality)

        return quality_index

    def get_inversion_index(self):
        """
        TODO
        """

        inversion_index = INVERSIONS.index(self.inversion)

        return inversion_index
