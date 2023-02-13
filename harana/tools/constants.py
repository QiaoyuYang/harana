import os
import numpy as np

__all__ = [
    'HOME',
    'ROOT_DIR',
    'DEFAULT_DATASETS_DIR',
    'DEFAULT_GENERATED_DIR',
    'DEFAULT_GROUND_TRUTH_DIR',
    'DEFAULT_CHECKPOINT_DIR',
    'NPZ_EXT',
    'CSV_EXT',
    'XLSX_EXT',
    'PT_EXT',
    'KEY_TRACK',
    'KEY_PC_ACT',
    'KEY_BASS_PC',
    'KEY_HARMONY_INDEX_GT',
    'KEY_HARMONY_COMPONENT_GT',
    'KEY_OFFSET',
    'KEY_METER',
    'TICKS_PER_QUARTER',
    'FRAMES_PER_QUARTER',
    'TICKS_PER_FRAME',
    'XLSX_EXT',
    'NUM_PC',
    'NUM_PIANO_KEYS',
    'MIN_MIDI',
    'C_OFFSET',
    'PC2PS',
    'PS2PC',
    'NUM_ROOTS',
    'NUM_QUALITIES',
    'QUALITIES',
    'CLEAN_QUALITIES',
    'NUM_INVERSIONS',
    'INVERSIONS',
    'NUM_TONICS',
    'CLEAN_TONICS',
    'NUM_MODES',
    'MODES',
    'NUM_KEYS',
    'NUM_PRI_DEGREES',
    'NUM_SEC_DEGREES',
    'SEC_DEG',
    'DEG2PC',
    'NUM_DEGREES',
    'HARMONY_TYPE_CHORD',
    'HARMONY_TYPE_RN',
    'HARMONY_COMPONENTS',
    'HARMONY_COMPONENT_DIMS',
    'CHORD_COMPONENT_DIMS',
    'RN_COMPONENT_DIMS',
    'NUM_HARMONY_COMPONENTS',
    'HARMONY_VEC_SIZE',
    'EMBEDDING_SIZE',
    'NUM_HARMONIES',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_SAMPLE_SIZE',
    'DEFAULT_MAX_SEG_LEN',
    'DEFAULT_HARMONY_TYPE',
    'DEFAULT_DEVICE',
    'DEFAULT_LR',
    'DEFAULT_MAX_EPOCH',
    'DEFAULT_EVAL_PERIOD',
    'STAGES'
]

##################################################
# PATHS                                          #
##################################################

HOME = os.path.expanduser('~')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

DEFAULT_DATASETS_DIR = os.path.join(HOME, 'Desktop', 'Datasets')

DEFAULT_GENERATED_DIR = os.path.join(ROOT_DIR, 'generated')
DEFAULT_GROUND_TRUTH_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'ground_truth')
DEFAULT_CHECKPOINT_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'checkpoint')

##################################################
# FILE EXTENSIONS                                #
##################################################

NPZ_EXT = 'npz'
CSV_EXT = 'csv'
XLSX_EXT = 'xlsx'
PT_EXT = 'pt'

##################################################
# DICTIONARY KEYS                                #
##################################################

KEY_TRACK = 'track'

KEY_PC_ACT = 'pitch_class_activity'
KEY_BASS_PC = 'bass_pc'

KEY_HARMONY_INDEX_GT = 'harmony_index_gt'
KEY_HARMONY_COMPONENT_GT = 'harmony_component_gt'

KEY_OFFSET = 'negative_frame_offset'
KEY_METER = 'meter'

##################################################
# TIME ATTRIBUTES                                #
##################################################

TICKS_PER_QUARTER = 24
FRAMES_PER_QUARTER = 4
TICKS_PER_FRAME = TICKS_PER_QUARTER / FRAMES_PER_QUARTER

##################################################
# NOTE ATTRIBUTES                                #
##################################################

NUM_PC = 12
NUM_PIANO_KEYS = 88
MIN_MIDI = 21
C_OFFSET = NUM_PC - MIN_MIDI % NUM_PC

PC2PS = {
    0  : 'C',
    1 : 'C♯',
    2 : 'D',
    3  : 'E♭',
    4 : 'E',
    6 : 'F♯',
    7  : 'G',
    8  : 'A♭',
    9 : 'A',
    10 : 'B♭',
    11  : 'B',
}

PS2PC = {
    'C♭' : 11,
    'C' : 0,
    'C♯' : 1,
    'D♭' : 1,
    'D' : 2,
    'D♯' : 3,
    'E♭' : 3,
    'E' : 4,
    'E♯' : 5,
    'F♭' : 4,
    'F' : 5,
    'F♯' : 6,
    'G♭' : 6,
    'G' : 7,
    'G♯' : 8,
    'A♭' : 8,
    'A' : 9,
    'A♯' : 10,
    'B♭' : 10,
    'B' : 11,
    'B♯' : 0,
}

##################################################
# HARMONY ATTRIBUTES                             #
##################################################

# ROOT
NUM_ROOTS = 12

# QUALITY
NUM_QUALITIES = 10
QUALITIES = ['maj', 'min', 'aug', 'dim', 'maj7', 'min7', 'dom7', 'hdi7', 'dim7', 'aug6']
CLEAN_QUALITIES = {
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

# INVERSION
NUM_INVERSIONS = 4
INVERSIONS = ['root', '1st', '2nd', '3rd']

# KEY
NUM_TONICS = 12
CLEAN_TONICS = {
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

NUM_MODES = 2
MODES = ['ionian', 'aeolian']

NUM_KEYS = NUM_TONICS * NUM_MODES
# 24 = 12 * 2

# DEGREE
NUM_PRI_DEGREES = 12
NUM_SEC_DEGREES = 3
SEC_DEG = ['1', '5', '7']

DEG2PC = {
    '-1' : 11,
    '1' : 0,
    '+1' : 1,
    '-2' : 1,
    '2' : 2,
    '+2' : 3,
    '-3' : 3,
    '3' : 4,
    '+3' : 5,
    '-4' : 4,
    '4' : 5,
    '+4' : 6,
    '-5' : 6,
    '5' : 7,
    '+5' : 8,
    '-6' : 8,
    '6' : 9,
    '+6' : 10,
    '-7' : 10,
    '7' : 11,
    '+7' : 0
}

NUM_DEGREES = NUM_PRI_DEGREES * NUM_SEC_DEGREES
# 36 = 12 * 3

HARMONY_TYPE_CHORD = 'CHORD'
HARMONY_TYPE_RN = 'RN'

HARMONY_COMPONENTS = {
    HARMONY_TYPE_CHORD :['root', 'quality', 'inversion'],
    HARMONY_TYPE_RN :['key', 'degree', 'quality', 'inversion']
}

COMPONENT_DIMS = {
    'root' : NUM_ROOTS,
    'quality' : NUM_QUALITIES,
    'inversion' : NUM_INVERSIONS,
    'key' : NUM_KEYS,
    'degree' : NUM_DEGREES
}

CHORD_COMPONENT_DIMS = [COMPONENT_DIMS[component] for component in HARMONY_COMPONENTS[HARMONY_TYPE_CHORD]]
RN_COMPONENT_DIMS = [COMPONENT_DIMS[component] for component in HARMONY_COMPONENTS[HARMONY_TYPE_RN]]

HARMONY_COMPONENT_DIMS = {
    HARMONY_TYPE_CHORD : CHORD_COMPONENT_DIMS,
    HARMONY_TYPE_RN : RN_COMPONENT_DIMS
}

NUM_HARMONY_COMPONENTS = {
    HARMONY_TYPE_CHORD : len(CHORD_COMPONENT_DIMS),
    HARMONY_TYPE_RN : len(RN_COMPONENT_DIMS)
}

HARMONY_VEC_SIZE = {
    HARMONY_TYPE_CHORD : np.sum(CHORD_COMPONENT_DIMS),
    # 26 = 12 + 10 + 4
    HARMONY_TYPE_RN : np.sum(RN_COMPONENT_DIMS)
    # 74 = 24 + 36 + 10 + 4
}

NUM_HARMONIES = {
    HARMONY_TYPE_CHORD : np.prod(CHORD_COMPONENT_DIMS),
    # 480 = 12 * 10 * 4
    HARMONY_TYPE_RN : np.prod(RN_COMPONENT_DIMS)
    # 34560 = 24 * 36 * 10 * 4
}

EMBEDDING_SIZE = {
    HARMONY_TYPE_CHORD : 10,
    # 10 = 4 + 4 + 2
    HARMONY_TYPE_RN : 16
    # 17 = 5 + 6 + 4 + 2
}

##################################################
# PIPELINE ATTRIBUTES                            #
##################################################

DEFAULT_BATCH_SIZE = 8
DEFAULT_SAMPLE_SIZE = 48
DEFAULT_MAX_SEG_LEN = 16
DEFAULT_HARMONY_TYPE = HARMONY_TYPE_CHORD
DEFAULT_DEVICE = 'cpu'
DEFAULT_LR = 0.001
DEFAULT_MAX_EPOCH = 1000
DEFAULT_EVAL_PERIOD = 100
STAGES = {"Validation", "Training"}
