import os

SIMPLE_ITK = 'SimpleITK'
NATURAL = 'NATURAL'
CT = 'CT'

RAW_ROOT = os.getenv('RAW_ROOT')
if RAW_ROOT is None or not os.path.exists(RAW_ROOT):
    raise NotADirectoryError('You must define $RAW_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), and make sure that the path exists.')

PREPROCESSED_ROOT = os.getenv('PREPROCESSED_ROOT')
if PREPROCESSED_ROOT is None or not os.path.exists(PREPROCESSED_ROOT):
    raise NotADirectoryError('You must define $PREPROCESSED_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), and make sure that the path exists.')

RESULTS_ROOT = os.getenv('RESULTS_ROOT')
if RESULTS_ROOT is None or not os.path.exists(RESULTS_ROOT):
    raise NotADirectoryError('You must define $RESULTS_ROOT in your environment variables '
                             '(in ~/.bashrc or ~/.profile), and make sure that the path exists.')


ENB6 = 'enb6'
ENB4 = 'enb4'
ENV2 = 'env2'
ENB0 = 'enb0'
ENB1 = "enb1"

CONCAT = 'concat'
ADD = 'add'
TWO_D = '2d'
THREE_D = '3d'
INSTANCE = 'instance'
BATCH = "batch"
BASE = 'base'
ECHO = 'echo'
IMAGENET = 'imagenet'
NUTRIENT = 'nutrient'
PATIENT_PATH = "patient_path"
FILE_NAME = "file_name"
INSTANCE_NUMBER = "InstanceNumber"
LABEL = "label"
ENDOCARDIAL_QUALITY = "endocardial_quality"
AXIS_QUALITY = "axis_quality"
STRUCTURE_QUALITY = "structure_quality"
QUALITY_SUM = "quality_sum"

SEGMENTATION = "segmentation"
CLASSIFICATION = "classification"
