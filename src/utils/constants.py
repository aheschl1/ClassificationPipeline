import os

SIMPLE_ITK = 'SimpleITK'
NATURAL = 'NATURAL'
CT = 'CT'
DATA_ROOT = os.getenv('DATASET_ROOT')
if DATA_ROOT is None or not os.path.exists(DATA_ROOT):
    raise NotADirectoryError('You must define $DATASET_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), and make sure that the path exists.')

RAW_ROOT = f"{DATA_ROOT}/raw"
PREPROCESSED_ROOT = f"{DATA_ROOT}/preprocessed"
RESULTS_ROOT = f"{DATA_ROOT}/results"
ENB6 = 'enb6'
CONCAT = 'concat'
ADD = 'add'
TWO_D = '2d'
THREE_D = '3d'
INSTANCE = 'instance'
BATCH = "batch"
