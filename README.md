# SegClass Pipeline
This pipeline is for semantic segmentation and classification.

# Supported file types
This project supports:
1. jpeg
2. jpg
3. png
4. nii.gz
For segmentation, mask and image must be same filetype.

# Getting started
To start, you must:
1. On src/training/training_entry.py and src/preprocessing/preprocess_entry.py add the line: sys.path.append('/my/path/to/ClassificationPipeline')
2. Add RAW_ROOT, RESULTS_ROOT, and PREPROCESSED_ROOT to your environent variables (in ~/.bashrc for example).
3. Run python training_entry.py --help for available option
4. Run python preprocess_entry.py --help for available options

# Before training
1. Run python preprocess_entry.py --help for available options
2. Preprocess your desired dataset. Make sure you format your data properly (described below) or use one of the custom preprocessors if it applies to you.

# Dataset structure classification
You must structure a dataset as follows:
RAW_ROOT/Dataset_XXX/labels_xxxx
Where RAW_ROOT is the path in your environment.
XXX is your dataset ID number. Must be three digits.
Last folder (labels_xxxx) shows that you must have a folder for each class name.
Place files in the corresponding label folders.

# Dataset structure segmentation
You must structure a dataset as follows:
RAW_ROOT/Dataset_XXX/imagesTr
RAW_ROOT/Dataset_XXX/labelsTr
Where RAW_ROOT is the path in your environment.
XXX is your dataset ID number. Must be three digits.
Place masks in labelsTr
Place images in imagesTr

# File names
All files must follow the structure case_xxxxx. ex: case_00001, case_89234
You must not overlap case names.
Match image and mask names for segmentation.
