# Our imports
from .constants import *

# Regular imports
import numpy as np

##################################################
# NOTE UTILS                                     #
##################################################

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

##################################################
# HARMONY UTILS                                  #
##################################################

class Chord:

    def __init__(self, root_pc, quality, inversion, onset, duration):
        
        self.root_pc = root_pc
        self.quality = quality
        self.inversion = inversion
        self.onset = onset
        self.duration = duration

    def get_index(self):
        
        quality_index = QUALITIES.index(self.quality)
        inversion_index = INVERSIONS.index(self.inversion)

        return inversion_index + quality_index * NUM_INVERSIONS + \
                self.root_pc * NUM_QUALITIES * NUM_INVERSIONS
    
    def get_offset(self):
        
        return self.onset + self.duration - 1
    
    def __repr__(self):
        
        self.root_ps = PC2PS[self.root_pc]
        
        return "Chord(root = {}, quality = {}, inversion = {}, onset = {}, duration = {})".\
            format(self.root_ps, self.quality, self.inversion, self.onset, self.duration)

class Key:

    def __init__(self, tonic_ps, mode):
        
        self.tonic_ps = tonic_ps
        self.mode = mode
        self.tonic_pc = PS2PC[self.tonic_ps]
        self.mode_index = MODES.index(mode)

    def get_index(self):
        
        return NUM_MODES * self.tonic_pc + self.mode_index

    @staticmethod
    def parse_index(key_index):
        
        mode_index = key_index % NUM_MODES
        tonic_pc = int((key_index - mode_index) / NUM_MODES)
        
        return int(tonic_pc), int(mode_index)

    def __repr__(self):
        
        return f"{self.tonic_ps}_{self.mode}"

class Degree:

    def __init__(self, degree_symbol):

        self.pri_deg, self.sec_deg = Degree.parse_symbol(degree_symbol)
        self.pri_deg_pc = DEG2PC[self.pri_deg]
        self.sec_deg_index = SEC_DEG.index(self.sec_deg)

    def get_index(self):
        
        return NUM_PRI_DEGREES * self.sec_deg_index + self.pri_deg_pc

    @staticmethod
    def parse_index(degree_index):
        
        pri_deg_pc = degree_index % NUM_PRI_DEGREES
        sec_deg_index = int((degree_index - pri_deg_pc) / NUM_PRI_DEGREES)
        
        return int(pri_deg_pc), int(sec_deg_index)

    @staticmethod
    def parse_symbol(degree_symbol):
        
        if "/" in degree_symbol:
            sec_deg, pri_deg = degree_symbol.split("/")
        else:
            pri_deg = degree_symbol
            sec_deg = "1"
        
        return pri_deg, sec_deg

    def __repr__(self):
        
        if self.sec_deg == "1":    
            return f"{self.pri_deg}"
        else:
            return f"{self.sec_deg}/{self.pri_deg}"

class RomanNumeral:

    def __init__(self, key, degree, quality, inversion, onset, duration):
        
        self.key = key
        self.degree = degree
        self.quality = quality
        self.inversion = inversion
        self.onset = onset
        self.duration = duration

    def get_index(self):
        
        key_index = self.key.get_index()
        degree_index = self.degree.get_index()
        quality_index = QUALITIES.index(self.quality)
        inversion_index = INVERSIONS.index(self.inversion)
        
        return inversion_index + quality_index * NUM_INVERSIONS + \
            degree_index * NUM_QUALITIES * NUM_INVERSIONS + key_index * NUM_DEGREES * NUM_QUALITIES * NUM_INVERSIONS

    def get_offset(self):
        
        return self.onset + self.duration - 1
    
    def __repr__(self):
        
        return "RomanNumeral(key = {}, degree = {}, quality = {}, inversion = {}, onset = {}, duration = {})".\
            format(self.key.__repr__(), self.degree.__repr__(), self.quality, self.inversion, self.onset, self.duration)

class Harmony:
    
    def __init__(self, key, degree, quality, inversion, onset, duration):
        
        self.key = key
        self.degree = degree
        self.quality = quality
        self.inversion = inversion
        self.onset = onset
        self.duration = duration

        self.chord = Chord(self.get_root_pc(), quality, inversion, onset, duration)
        self.rn = RomanNumeral(key, degree, quality, inversion, onset, duration)

    def get_root_pc(self):
        
        tonic_pc = self.key.tonic_pc
        pri_deg_pc = self.degree.pri_deg_pc
        sec_deg_pc = DEG2PC[self.degree.sec_deg]
        root_pc = (tonic_pc + pri_deg_pc + sec_deg_pc)%12

        return root_pc

    def get_offset(self):
        
        return self.onset + self.duration - 1
    
    @staticmethod
    def parse_harmony_index(harmony_index, harmony_type):
        
        if harmony_type == "CHORD":
            
            root_pc = np.floor(harmony_index / (NUM_QUALITIES * NUM_INVERSIONS))
            
            index_remain = harmony_index - root_pc * NUM_QUALITIES * NUM_INVERSIONS
            quality_index = np.floor(index_remain / NUM_INVERSIONS)
            
            index_remain -= quality_index * NUM_INVERSIONS
            inversion_index = index_remain
            
            return int(root_pc), int(quality_index), int(inversion_index)
        
        elif harmony_type == "RN":
            
            key_index = np.floor(harmony_index / (NUM_DEGREES * NUM_QUALITIES * NUM_INVERSIONS))
            
            index_remain = harmony_index - key_index * NUM_DEGREES * NUM_QUALITIES * NUM_INVERSIONS
            degree_index = np.floor(index_remain / (NUM_QUALITIES * NUM_INVERSIONS))
            
            index_remain -= degree_index * NUM_QUALITIES * NUM_INVERSIONS
            quality_index = np.floor(index_remain / NUM_INVERSIONS)
            
            index_remain -= quality_index * NUM_INVERSIONS
            inversion_index = index_remain

            return int(key_index), int(degree_index), int(quality_index), int(inversion_index)

##################################################
# RHYTHM UTILS                                   #
##################################################

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
