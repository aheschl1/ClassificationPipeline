import os

SIMPLE_ITK = 'SimpleITK'
DATA_ROOT = os.getenv('DATASET_ROOT')

if DATA_ROOT is None or not os.path.exists(DATA_ROOT):
    raise NotADirectoryError('You must define $DATASET_ROOT in your environment variables (in .bashrc), and make sure '
                             'that the path exists.')

RAW_ROOT = f"{DATA_ROOT}/raw"
PREPROCESSED_ROOT = f"{DATA_ROOT}/preprocessed"
