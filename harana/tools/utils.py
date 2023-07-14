# Our imports
from .constants import *

# Regular imports
import numpy as np
import torch.nn as nn

##################################################
# NOTE UTILS                                     #
##################################################

class Note(object):
    """
    TODO

    TODO - all time here is in ticks
    """

    def __init__(self, midi_num, onset, duration):
        """
        TODO
        """

        self.midi_num = midi_num
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
        key_index = self.midi_num - MIN_MIDI
        if key_index >= NUM_PIANO_KEYS:
            key_index -= NUM_PC

        return key_index

    def get_pitch_class_index(self):
        """
        TODO
        """

        # Offset to start at C and obtain remainder
        pitch_class_index = (self.get_key_index() - C_OFFSET) % NUM_PC

        return pitch_class_index

    def __repr__(self):
        return f"Note(pitch_name={PC2PS[self.get_pitch_class_index()]}, onset={self.onset}, duration={self.duration})"

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

        return inversion_index + quality_index * NUM_INVERSIONS + self.root_pc * NUM_QUALITIES * NUM_INVERSIONS

    @staticmethod
    def get_chordal_pc(root_pc, chord_index):
        
        chordal_relative_pc = QUALITY2PC[QUALITIES[chord_index]]
        chordal_pc = [(root_pc + relative_pc) % NUM_PC for relative_pc in chordal_relative_pc]
        
        return chordal_pc
    
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
        
        return self.mode_index + self.tonic_pc * NUM_MODES

    def shift_tonic(self, tonic_shift):
        self.tonic_pc = (self.tonic_pc + tonic_shift) % NUM_PC
        self.tonic_ps = PC2PS[self.tonic_pc]

    @staticmethod
    def parse_index(key_index):
        
        tonic_pc = np.floor(key_index / NUM_MODES)
        mode_index = key_index - tonic_pc * NUM_MODES
        
        return int(tonic_pc), int(mode_index)

    @staticmethod
    def get_mode_pc(key_index):
        
        tonic_pc, mode_index = Key.parse_index(key_index)
        mode_pc = (tonic_pc + np.array(MODE_RELATIVE_PC[MODES[mode_index]])) % NUM_PC
        
        return mode_pc


    def __repr__(self):
        
        return f"{self.tonic_ps}_{self.mode}"

class Degree:

    def __init__(self, degree_symbol):

        self.pri_deg, self.sec_deg = Degree.parse_symbol(degree_symbol)
        self.pri_deg_pc = DEG2PC[self.pri_deg]
        self.sec_deg_pc = DEG2PC[self.sec_deg]

    def get_index(self):
        
        return NUM_PRI_DEGREES * self.sec_deg_pc + self.pri_deg_pc

    @staticmethod
    def parse_index(degree_index):
        
        pri_deg_pc = degree_index % NUM_PRI_DEGREES
        sec_deg_index = int((degree_index - pri_deg_pc) / NUM_PRI_DEGREES)
        
        return int(pri_deg_pc), int(sec_deg_index)

    @staticmethod
    def parse_symbol(degree_symbol):
        
        if "/" in degree_symbol:
            sec_deg, pri_deg = degree_symbol.split("/")
            if pri_deg == '1':
                pri_deg = sec_deg
                sec_deg = '1'
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

    def get_rq_index(self):
        root_pc = self.get_root_pc()
        quality_index = QUALITIES.index(self.quality)
        
        return quality_index + root_pc * NUM_QUALITIES

    def get_krq_index(self):
        key_index = self.key.get_index()
        root_pc = self.get_root_pc()
        quality_index = QUALITIES.index(self.quality)
        
        return quality_index + root_pc * NUM_QUALITIES + key_index * NUM_PC * NUM_QUALITIES

    @staticmethod
    def index2symbol(harmony_index, harmony_type):
        if harmony_type == "RQ":
            root_pc, quality_index = Harmony.parse_harmony_index(harmony_index, harmony_type)
            return f"{PC2PS[root_pc]}:{QUALITIES[quality_index]}"

    @staticmethod
    def combine_component_indexes(component_indexes, harmony_type):
        if harmony_type == "RQ":
            root_pc, quality_index = component_indexes
            return quality_index + root_pc * NUM_QUALITIES
    
    @staticmethod
    def parse_harmony_index(harmony_index, harmony_type):

        if harmony_type == "K":

            key_index = harmony_index
            
            return [int(key_index)]
        
        if harmony_type == "RQ":
            
            root_pc = np.floor(harmony_index / (NUM_QUALITIES))
            quality_index = harmony_index - root_pc * NUM_QUALITIES

            return [int(root_pc), int(quality_index)]

        elif harmony_type == "KRQ":
            
            key_index = np.floor(harmony_index / (NUM_PC * NUM_QUALITIES))
            
            index_remain = harmony_index - key_index * NUM_PC * NUM_QUALITIES
            root_pc = np.floor(index_remain / (NUM_QUALITIES))
            
            index_remain -= root_pc * NUM_QUALITIES
            quality_index = index_remain

            return [int(key_index), int(root_pc), int(quality_index)]

##################################################
# RHYTHM UTILS                                   #
##################################################

class Meter(object):
    """
    TODO
    """

    def __init__(self, beat_count, division, start_measure, end_measure):
        """
        TODO
        """

        self.beat_count = beat_count
        self.division = division
        self.start_measure = start_measure
        self.end_measure = end_measure
        self.num_measures = end_measure - start_measure + 1


    def get_measure_length(self):
        """
        TODO
        """

        # Compute the length of a measure in quarter notes using metric information
        measure_length = round(self.beat_count * (4 / self.division))

        return measure_length

    def get_quarters_per_beat(self):

        return round(self.get_measure_length()) / self.beat_count

    def __repr__(self):
        return f"Meter(time_signature = {self.beat_count}/{self.division}, start_measure = {self.start_measure}, end_measure = {self.end_measure})"

##################################################
# MODEL UTILS                                    #
##################################################

class LinearLayers(nn.Module):

    def __init__(self, num_layers, input_size, output_size):

        super().__init__()
        
        self.num_layers = num_layers
        hidden_size = round((input_size + output_size) / 2)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_size, hidden_size))
        for i in range(self.num_layers - 2):
            self.linears.append(nn.Linear(hidden_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))

        self.lns = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_size))
        self.lns.append(nn.LayerNorm(output_size))

        self.lkrelu = nn.LeakyReLU(inplace=True)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        
        out = self.lkrelu(self.lns[0](self.linears[0](x)))
        
        for i in range(self.num_layers - 2):
            out = self.lkrelu(self.lns[i](self.linears[i](out)))
            out = self.dropout(out)
        
        if self.num_layers > 1:
            out = self.lkrelu(self.lns[-1](self.linears[-1](out)))

        return out
