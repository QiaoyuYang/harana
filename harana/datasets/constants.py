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
    'XLSX_EXT'
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
