import os

__all__ = [
    'HOME',
    'ROOT_DIR',
    'DEFAULT_DATASETS_DIR',
    'DEFAULT_GENERATED_DIR',
    'DEFAULT_GROUND_TRUTH_DIR',
    'NPZ_EXT',
    'CSV_EXT',
    'KEY_TRACK',
    'KEY_PC_ACT',
    'KEY_PC_DST',
    'KEY_OFFSET',
    'KEY_METER',
    'XLSX_EXT',
    'NUM_PC',
    'NUM_KEYS',
    'MIN_MIDI',
    'C_OFFSET',
    'TONICS',
    'CHORD_QUALITIES',
    'INVERSIONS',
    'MODES',
    'ACCIDENTALS',
    'FUNCTIONS'
]

##################################################
# PATHS                                          #
##################################################

HOME = os.path.expanduser('~')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

DEFAULT_DATASETS_DIR = os.path.join(HOME, 'Desktop', 'Datasets')

DEFAULT_GENERATED_DIR = os.path.join(ROOT_DIR, 'generated')
DEFAULT_GROUND_TRUTH_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'ground_truth')

##################################################
# FILE EXTENSIONS                                #
##################################################

NPZ_EXT = 'npz'
CSV_EXT = 'csv'
XLSX_EXT = 'xlsx'

##################################################
# DICTIONARY KEYS                                #
##################################################

KEY_TRACK = 'track'
KEY_PC_ACT = 'pitch_class_activity'
KEY_PC_DST = 'pitch_class_distr'
KEY_OFFSET = 'negative_frame_offset'
KEY_METER = 'meter'

##################################################
# MUSICAL ATTRIBUTES                             #
##################################################

NUM_PC = 12
NUM_KEYS = 88
MIN_MIDI = 21
C_OFFSET = NUM_PC - MIN_MIDI % NUM_PC

##################################################
# MUSICAL THEORY                                 #
##################################################

# TODO - should this include B♯/C♭ or E♯/F♭?
TONICS = {
    'C'  : 'C',
    'C+' : 'C♯',
    'D-' : 'D♭',
    'D'  : 'D',
    'D+' : 'D♯',
    'E-' : 'E♭',
    'E'  : 'E',
    'F'  : 'F',
    'F+' : 'F♯',
    'G-' : 'G♭',
    'G'  : 'G',
    'G+' : 'G♯',
    'A-' : 'A♭',
    'A'  : 'A',
    'A+' : 'A♯',
    'B-' : 'B♭',
    'B'  : 'B'
    }

CHORD_QUALITIES = {
    'M': 'maj',
    'm': 'min',
    'a': 'aug',
    'd': 'dim',
    'M7': 'maj7',
    'm7': 'min7',
    'D7': 'dom7',
    'h7': 'hdi7',
    'd7': 'dim7',
    'a6': 'aug6'
}

INVERSIONS = ['root', '1st', '2nd', '3rd']

MODES = ['ionian', 'aeolian']

ACCIDENTALS = ['♮', '♭', '♯']

FUNCTIONS = {
    '1'  : 0,
    '♭2' : 1,
    '2'  : 2,
    '♭3' : 3,
    '3'  : 4,
    '4'  : 5,
    '♯4' : 6,
    '♭5' : 6,
    '5'  : 7,
    '♭6' : 8,
    '6'  : 9,
    '♭7' : 10,
    '7'  : 11
}
