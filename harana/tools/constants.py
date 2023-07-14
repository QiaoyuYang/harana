import os
import numpy as np

__all__ = [
    'HOME',
    'ROOT_DIR',
    'DEFAULT_DATASETS_DIR',
    'DEFAULT_GENERATED_DIR',
    'DEFAULT_GROUND_TRUTH_DIR',
    'DEFAULT_CHECKPOINT_DIR',
    'NUM_TRACKS',
    'TXT_EXT',
    'NPY_EXT',
    'NPZ_EXT',
    'CSV_EXT',
    'XLSX_EXT',
    'MXL_EXT',
    'PT_EXT',
    'KEY_TRACK',
    'KEY_PC_ACT',
    'KEY_BASS_PC',
    'KEY_HARMONY_INDEX_GT',
    'KEY_HARMONY_COMPONENT_GT',
    'HARMONY_KEYS',
    'KEY_OFFSET',
    'KEY_METER',
    'NON_ARRAY_KEYS',
    'TICKS_PER_QUARTER',
    'FRAMES_PER_QUARTER',
    'TICKS_PER_BEAT',
    'FRAMES_PER_BEAT',
    'TICKS_PER_FRAME',
    'HARMONY_UNIT_SPAN',
    'XLSX_EXT',
    'NOTE_MAP',
    'NOTE_TYPE_MAP',
    'NUM_PC',
    'NUM_MIDI',
    'NUM_PIANO_KEYS',
    'MIN_MIDI',
    'C_OFFSET',
    'PC2PS',
    'PS2PC',
    'NUM_ROOTS',
    'NUM_QUALITIES',
    'QUALITIES',
    'QUALITY2PC',
    'QUALITY2MAJMIN',
    'CLEAN_QUALITIES',
    'NUM_INVERSIONS',
    'INVERSIONS',
    'NUM_TONICS',
    'CLEAN_TONICS',
    'NUM_MODES',
    'MODES',
    'MODE_RELATIVE_PC',
    'NUM_KEYS',
    'NUM_PRI_DEGREES',
    'NUM_SEC_DEGREES',
    'DEG2PC',
    'NUM_DEGREES',
    'HARMONY_TYPE_K',
    'HARMONY_TYPE_RQ',
    'HARMONY_TYPE_KRQ',
    'HARMONY_COMPONENTS',
    'HARMONY_COMPONENT_DIMS',
    'K_COMPONENT_DIMS',
    'RQ_COMPONENT_DIMS',
    'KRQ_COMPONENT_DIMS',
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
    'PIPELINE_TYPE_TRAIN',
    'PIPELINE_TYPE_TEST',
    'STAGES'
]

##################################################
# FILES                                          #
##################################################

HOME = os.path.expanduser('~')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

DEFAULT_DATASETS_DIR = os.path.join(ROOT_DIR, 'dataset')

DEFAULT_GENERATED_DIR = os.path.join(ROOT_DIR, 'generated')
DEFAULT_GROUND_TRUTH_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'ground_truth')
DEFAULT_CHECKPOINT_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'checkpoint')

BPSFH_NUM_TRACKS = 32
TAVERN_NUM_TRACKS = 17 + 10
ROMANTXT_NUM_TRACKS = 48 + 24 + 70 + 48
LOPEZ_NUM_TRACKS = 24 + 156
OTHER_NUM_TRACKS = 30

NUM_TRACKS = BPSFH_NUM_TRACKS + TAVERN_NUM_TRACKS + ROMANTXT_NUM_TRACKS + LOPEZ_NUM_TRACKS + OTHER_NUM_TRACKS
#NUM_TRACKS = 32

##################################################
# FILE EXTENSIONS                                #
##################################################

TXT_EXT = 'txt'
NPY_EXT = 'npy'
NPZ_EXT = 'npz'
CSV_EXT = 'csv'
XLSX_EXT = 'xlsx'
MXL_EXT = 'mxl'
PT_EXT = 'pt'

##################################################
# DICTIONARY KEYS                                #
##################################################

KEY_TRACK = 'track'

KEY_PC_ACT = 'pitch_class_activity'
KEY_BASS_PC = 'bass_pc'

KEY_HARMONY_INDEX_GT = 'harmony_index_gt'
KEY_HARMONY_COMPONENT_GT = 'harmony_component_gt'

HARMONY_KEYS = [KEY_HARMONY_INDEX_GT, KEY_HARMONY_COMPONENT_GT]

KEY_OFFSET = 'negative_frame_offset'
KEY_METER = 'meter'

NON_ARRAY_KEYS = [KEY_OFFSET, KEY_METER]

##################################################
# TIME ATTRIBUTES                                #
##################################################

TICKS_PER_QUARTER = 24
FRAMES_PER_QUARTER = 8
TICKS_PER_BEAT = 24
FRAMES_PER_BEAT = 8
TICKS_PER_FRAME = TICKS_PER_BEAT / FRAMES_PER_BEAT
HARMONY_UNIT_SPAN = 4


##################################################
# NOTE ATTRIBUTES                                #
##################################################

NOTE_MAP = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

NOTE_TYPE_MAP = {
    "1024th": 0.00390625,
    "512th": 0.0078125,
    "256th": 0.015625,
    "128th": 0.03125,
    "64th": 0.0625,
    "32nd": 0.125,
    "16th": 0.25,
    "eighth": 0.5,
    "quarter": 1.0,
    "half": 2.0,
    "whole": 4.0,
    "breve": 8.0,
    "long": 16.0,
    "maxima": 32.0,
}

NUM_PC = 12
NUM_PIANO_KEYS = 88
NUM_MIDI = 127
MIN_MIDI = 21
C_OFFSET = NUM_PC - MIN_MIDI % NUM_PC

PC2PS = {
    0  : 'C',
    1 : 'C#',
    2 : 'D',
    3  : 'Eb',
    4 : 'E',
    5 : 'F',
    6 : 'F#',
    7  : 'G',
    8  : 'Ab',
    9 : 'A',
    10 : 'Bb',
    11  : 'B',
}

PS2PC = {
    'Cb' : 11,
    'C' : 0,
    'C#' : 1,
    'Db' : 1,
    'D' : 2,
    'D#' : 3,
    'Eb' : 3,
    'E' : 4,
    'E#' : 5,
    'Fb' : 4,
    'F' : 5,
    'F#' : 6,
    'Gb' : 6,
    'G' : 7,
    'G#' : 8,
    'Ab' : 8,
    'A' : 9,
    'A#' : 10,
    'Bb' : 10,
    'B' : 11,
    'B#' : 0,
}

##################################################
# HARMONY ATTRIBUTES                             #
##################################################

# ROOT
NUM_ROOTS = 12

# QUALITY
NUM_QUALITIES = 10
QUALITIES = ['maj', 'min', 'aug', 'dim', 'maj7', 'min7', 'dom7', 'hdi7', 'dim7', 'aug6']
QUALITY2PC = {
    'maj' : [0, 4, 7],
    'min' : [0, 3, 7],
    'aug' : [0, 4, 8],
    'dim' : [0, 3, 6],
    'maj7' : [0, 4, 7, 11],
    'min7' : [0, 3, 7, 10],
    'dom7' : [0, 4, 7, 10],
    'hdi7' : [0, 3, 6, 10],
    'dim7' : [0, 3, 6, 9],
    'aug6' : [0, 2, 6]
}

QUALITY2MAJMIN = {
    0:0,
    1:1,
    2:0,
    3:1,
    4:0,
    5:1,
    6:0,
    7:1,
    8:1,
    9:0,
}

CLEAN_QUALITIES = {
    'M' : 'maj',
    'm' : 'min',
    'a' : 'aug',
    'd' : 'dim',
    'M7' : 'maj7',
    'm7' : 'min7',
    'D7' : 'dom7',
    'h7' : 'hdi7',
    'd7' : 'dim7',
    'a6' : 'aug6',
    'It+6' : 'aug6',
    'Gr+6' : 'aug6',
    'Fr+6' : 'aug6',
    'sus' : 'maj',
    'V4' : 'maj',
    'I4' : 'maj',
    'IV732' : 'maj',
    'IV765' : 'maj',
    'I532' : 'maj',
    'V6432' : 'maj',
    'bII6#5' : 'maj',
    'v4' : 'min',
    'iv4' : 'min',
    'viio62' : 'dim',
    'viio64b32' : 'dim',
    '+7' : 'maj7',
    'I7+6' : 'dom7',
    'V654' : 'dom7',
    'viøb7653' : 'hdi7',
    'viio742' : 'dim7'
}

# INVERSION
NUM_INVERSIONS = 4
INVERSIONS = ['root', '1st', '2nd', '3rd']

# KEY
NUM_TONICS = 12
CLEAN_TONICS = {
    'C-' : 'Cb',
    'C'  : 'C',
    'C+' : 'C#',
    'D-' : 'Db',
    'D'  : 'D',
    'D+' : 'D#',
    'E-' : 'Eb',
    'E'  : 'E',
    'E+' : 'E#',
    'F-' : 'Fb',
    'F'  : 'F',
    'F+' : 'F#',
    'G-' : 'Gb',
    'G'  : 'G',
    'G+' : 'G#',
    'A-' : 'Ab',
    'A'  : 'A',
    'A+' : 'A#',
    'B-' : 'Bb',
    'B'  : 'B',
    'B+' : 'B#'
    }

NUM_MODES = 2
MODES = ['ionian', 'aeolian']

MODE_RELATIVE_PC = {
    'ionian' : [0, 2, 4, 5, 7, 9, 11],
    'aeolian' : [0, 2, 3, 5, 7, 8, 10]
}

NUM_KEYS = NUM_TONICS * NUM_MODES
# 24 = 12 * 2

# DEGREE
NUM_PRI_DEGREES = 12
NUM_SEC_DEGREES = 12

DEG2PC = {
    '--1' : 10,
    '-1' : 11,
    '1' : 0,
    '+1' : 1,
    '++1' : 2,
    '--2' : 0,
    '-2' : 1,
    '2' : 2,
    '+2' : 3,
    '++2' : 4,
    '--3' : 2,
    '-3' : 3,
    '3' : 4,
    '+3' : 5,
    '++3' : 6,
    '--4' : 3,
    '-4' : 4,
    '4' : 5,
    '+4' : 6,
    '++4' : 7,
    '--5' : 5,
    '-5' : 6,
    '5' : 7,
    '+5' : 8,
    '++5' : 9,
    '--6' : 7,
    '-6' : 8,
    '6' : 9,
    '+6' : 10,
    '++6' : 11,
    '--7' : 9,
    '-7' : 10,
    '7' : 11,
    '+7' : 0,
    '++7' : 1,
}

NUM_DEGREES = NUM_PRI_DEGREES * NUM_SEC_DEGREES
# 36 = 12 * 3

HARMONY_TYPE_K = 'K'
HARMONY_TYPE_RQ = 'RQ'
HARMONY_TYPE_KRQ = 'KRQ'

HARMONY_COMPONENTS = {
    HARMONY_TYPE_K : ['key'],
    HARMONY_TYPE_RQ :['root', 'quality'],
    HARMONY_TYPE_KRQ :['key', 'root', 'quality'],
}

COMPONENT_DIMS = {
    'root' : NUM_ROOTS,
    'quality' : NUM_QUALITIES,
    'inversion' : NUM_INVERSIONS,
    'key' : NUM_KEYS,
    'degree' : NUM_DEGREES
}

K_COMPONENT_DIMS = [COMPONENT_DIMS[component] for component in HARMONY_COMPONENTS[HARMONY_TYPE_K]]
RQ_COMPONENT_DIMS = [COMPONENT_DIMS[component] for component in HARMONY_COMPONENTS[HARMONY_TYPE_RQ]]
KRQ_COMPONENT_DIMS = [COMPONENT_DIMS[component] for component in HARMONY_COMPONENTS[HARMONY_TYPE_KRQ]]

HARMONY_COMPONENT_DIMS = {
    HARMONY_TYPE_K : K_COMPONENT_DIMS,
    HARMONY_TYPE_RQ : RQ_COMPONENT_DIMS,
    HARMONY_TYPE_KRQ : KRQ_COMPONENT_DIMS
}

NUM_HARMONY_COMPONENTS = {
    HARMONY_TYPE_K : len(K_COMPONENT_DIMS),
    HARMONY_TYPE_RQ : len(RQ_COMPONENT_DIMS),
    HARMONY_TYPE_KRQ : len(KRQ_COMPONENT_DIMS),
}

HARMONY_VEC_SIZE = {
    HARMONY_TYPE_K : np.sum(K_COMPONENT_DIMS),
    # 24
    HARMONY_TYPE_RQ : np.sum(RQ_COMPONENT_DIMS),
    # 22 = 12 + 10
    HARMONY_TYPE_KRQ : np.sum(KRQ_COMPONENT_DIMS),
    # 46 = 24 + 12 + 10
}

NUM_HARMONIES = {
    HARMONY_TYPE_K : np.prod(K_COMPONENT_DIMS),
    # 24
    HARMONY_TYPE_RQ : np.prod(RQ_COMPONENT_DIMS),
    # 480 = 12 * 10 * 4
    HARMONY_TYPE_KRQ : np.prod(KRQ_COMPONENT_DIMS),
    # 2880 = 24 * 12 * 10
}

EMBEDDING_SIZE = 128

##################################################
# PIPELINE ATTRIBUTES                            #
##################################################


DEFAULT_BATCH_SIZE = 8

# in frames
DEFAULT_SAMPLE_SIZE = 96

# in beats
DEFAULT_MAX_SEG_LEN = 12

DEFAULT_HARMONY_TYPE = HARMONY_TYPE_RQ
DEFAULT_ENCODER_TYPE = 'CRNN'
DEFAULT_DECODER_TYPE = 'SemiCRF'
DEFAULT_DEVICE = 'cpu'
DEFAULT_LR = 0.001
DEFAULT_MAX_EPOCH = 1000
DEFAULT_EVAL_PERIOD = 100
PIPELINE_TYPE_TRAIN = 'train'
PIPELINE_TYPE_TEST = 'test'
STAGES = {
    PIPELINE_TYPE_TRAIN : ["Validation", "Training"],
    PIPELINE_TYPE_TEST : ['Test']
}
