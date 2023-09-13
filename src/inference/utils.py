import glob
import json
import os
from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, Resize

from src.dataloading.datapoint import Datapoint
from src.dataloading.dataset import PipelineDataset
from src.utils.utils import read_json
from src.utils.constants import PREPROCESSED_ROOT, RAW_ROOT


def get_dataset_from_folder(folder_path: str, dataset_name: str, fold: int, config: dict) -> PipelineDataset:
    """
    Prepares a dataset to run inference.
    :param folder_path: The path to the data
    :param dataset_name: The name of the dataset to use
    :param fold: The fold
    :param config: The config to use
    :return: Dataset
    """
    assert os.path.exists(folder_path), f"The data path {folder_path} does not exist :("
    files = glob.glob(f"{folder_path}/*")
    assert len(files) > 0, f"Did not find any files in {folder_path}. womp womp"
    datapoints = []
    try:
        mean_data = read_json(f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/mean_std.json")
    except FileNotFoundError:
        mean_data = None
    for file in files:
        datapoints.append(Datapoint(file, 0, dataset_name))
    transform_list = [
        Resize(config['target_size'], antialias=True)
    ]
    if mean_data is not None:
        transform_list.append(Normalize(mean=mean_data['mean'], std=mean_data['std']))
    transforms = Compose(transform_list)
    return PipelineDataset(datapoints, transforms)

