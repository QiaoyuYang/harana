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
        # TODO - should onset/offset ticks overlap or no?
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


class Chord(object):
    """
    TODO

    TODO - all time here is in ticks
    """

    def __init__(self, degree, quality, inversion, key, onset, duration, roman_numeral):
        """
        TODO
        """

        self.degree = degree
        self.quality = quality
        self.inversion = inversion
        self.key = key
        self.onset = onset
        self.duration = duration
        self.roman_numeral = roman_numeral

    def get_offset(self):
        """
        TODO
        """

        # Compute the last tick where chord is active
        # TODO - should onset/offset ticks overlap or no?
        offset = self.onset + self.duration - 1

        return offset
