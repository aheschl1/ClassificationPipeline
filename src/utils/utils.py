import json
from typing import Dict, Type, List, Union, Tuple
import os

import torch

from src.utils.constants import DATA_ROOT, PREPROCESSED_ROOT, RAW_ROOT
from torch.utils.data import DataLoader
import glob
from src.dataloading.datapoint import Datapoint
from src.dataloading.dataset import PipelineDataset


def write_json(data: Dict, path: str, create_folder: bool = False) -> None:
    """
    Write helper for json.
    :param data: Dictionary data to be written.
    :param path: The path to write.
    :param create_folder: If the path doesn't exist, should we create folders?
    :return: None
    """
    if not os.path.exists('/'.join(path.split('/')[0:-1])):
        assert create_folder, 'Path does not exist, and you did not indicate create_folder.'
        os.makedirs(path)

    with open(path, 'w') as file:
        json.dump(data, file)


def read_json(path: str) -> Dict:
    """
    Read json file.
    :param path:
    :return: Dictionary data of json file.
    """
    with open(path, 'r') as file:
        return json.load(file)


def get_dataset_name_from_id(id: Union[str, int]) -> str:
    """
    Given a dataset if that could be xxx or xx or x. Formats dataset id into dataset name.
    :param id: The dataset id
    :return: The dataset name.
    """
    if isinstance(id, int):
        id = str(id)
    if len(id) != 3:
        id = '0' * (3 - len(id)) + id
    return f"Dataset_{id}"


def check_raw_exists(dataset_name: str) -> bool:
    """
    Checks if the raw folder for a given dataset exists.
    :param dataset_name: The name of the dataset to check.
    :return: True if the raw folder exists, False otherwise.
    """
    assert "Dataset_" in dataset_name, f"You passed {dataset_name} to utils/check_raw_exists. Expected a dataset " \
                                       f"folder name."
    return os.path.exists(f"{RAW_ROOT}/{dataset_name}")


def verify_case_name(case_name: str) -> None:
    """
    Verifies that a case is named appropriately.
    If the case is named wrong, crashes the program.
    :param case_name: The name to check.
    :return: None
    """
    assert 'case_' in case_name, f"Invalid case name {case_name} in one of your folders. Case name " \
                                 "should be format case_xxxxx."
    assert len(case_name.split('_')[-1]) == 5, f"Invalid case name {case_name} in one of your folders. Case name " \
                                               "should be format case_xxxxx."


def get_raw_datapoints(dataset_name: str, label_to_id_mapping: Dict[str, int]) -> List[Datapoint]:
    """
    Given the name of a dataset, gets a list of datapoint objects.
    :param dataset_name: The name of the dataset.
    :param label_to_id_mapping: The label to id mapping for converting label name to number.
    :return: List of datapoints in the dataset.
    """
    dataset_root = f"{RAW_ROOT}/{dataset_name}/*/**"
    sample_paths = glob.glob(dataset_root)
    datapoints = []
    for path in sample_paths:
        label = path.split('/')[-2]
        label = label_to_id_mapping[label]
        name = path.split('/')[-1].split('.')[0]
        verify_case_name(name)
        datapoints.append(Datapoint(path, label, case_name=name, dataset_name=dataset_name))
    return datapoints


def get_preprocessed_datapoints(dataset_name: str, label_to_id_mapping: Dict[str, int], fold: int) \
        -> Tuple[List[Datapoint], List[Datapoint]]:
    """
    Returns the datapoints of preprocessed cases.
    :param dataset_name:
    :param label_to_id_mapping:
    :param fold:
    :return: Train points, Val points.
    """
    train_root = f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/*"
    train_paths = glob.glob(train_root)
    val_root = f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/*"
    val_paths = glob.glob(val_root)
    sample_paths = val_paths + train_paths

    train_datapoints, val_datapoints = [], []
    for path in sample_paths:
        label = path.split('/')[-2]
        label = label_to_id_mapping[label]
        name = path.split('/')[-1].split('.')[0]
        verify_case_name(name)
        if path in val_paths:
            val_datapoints.append(Datapoint(path, label, case_name=name, dataset_name=dataset_name))
        else:
            train_datapoints.append(Datapoint(path, label, case_name=name, dataset_name=dataset_name))
    return train_datapoints, val_datapoints


def get_raw_datapoints_folded(dataset_name: str, fold: int) -> Tuple[List[Datapoint], List[Datapoint]]:
    """
    Given a dataset name, returns the train and val points given a fold.
    :param dataset_name: The name of the dataset.
    :param fold: The fold to fetch.
    :return: Train and val points.
    """
    fold = get_folds_from_dataset(dataset_name)[str(fold)]

    datapoints = get_raw_datapoints(dataset_name, get_label_mapping_from_dataset(dataset_name))
    train_points, val_points = [], []
    # Now we populate the train and val lists
    for point in datapoints:
        if point.case_name in fold['train']:
            train_points.append(point)
        elif point.case_name in fold['val']:
            val_points.append(point)
        else:
            raise SystemError(f'{point.case_name} was not found in either the train or val fold! Maybe rerun '
                              f'preprocessing.')
    return train_points, val_points


def get_label_mapping_from_dataset(dataset_name: str, return_inverse: bool = False) -> Union[
    Tuple[Dict[str, int], Dict[int, str]], Dict[str, int]]:
    """
    Given a dataset name gets the label mapping from the preprocessed folder.
    :param dataset_name: The name of the dataset of interest.
    :param return_inverse: Whether to also return inverse mapping.
    :return: The label mapping requested.
    """
    label_to_id = f"{PREPROCESSED_ROOT}/{dataset_name}/label_to_id.json"
    id_to_label = f"{PREPROCESSED_ROOT}/{dataset_name}/id_to_label.json"
    label_to_id = read_json(label_to_id)
    id_to_label = read_json(id_to_label)
    if return_inverse:
        return label_to_id, id_to_label
    return label_to_id


def get_folds_from_dataset(dataset_name: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Fetches and loads the fold json from a dataset.
    :param dataset_name: The name of the dataset.
    :return: The fold dictionary.
    """
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/folds.json"
    return read_json(path)


def get_config_from_dataset(dataset_name: str) -> Dict:
    """
    Given a dataset name looks for a config file.
    :param dataset_name: The name of the dataset.
    :return: Config dictionary.
    """
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/config.json"
    return read_json(path)


def batch_collate_fn(batch: List[Datapoint]) -> Tuple[torch.Tensor, torch.Tensor, List[Datapoint]]:
    """
    Combines data fetched by dataloader into proper format.
    :param batch: List of data points from loader.
    :return: Batched tensor data, labels, and list of datapoints.
    """
    data = []
    labels = []
    for point in batch:
        data.append(point.data)
        labels.append(torch.tensor(point.label))
    return torch.stack(data), torch.stack(labels), batch


def get_dataloaders_from_fold(dataset_name: str, fold: int,
                              train_transforms=None, val_transforms=None,
                              preprocessed_data: bool = True,
                              store_metadata: bool = False,
                              **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the train and val dataloaders for a specific dataset fold.
    :param dataset_name: The name of the dataset.
    :param fold: The fold to grab.
    :param train_transforms: The transforms to apply to training data.
    :param val_transforms: The transforms to apply to val data.
    :param preprocessed_data: If true, grabs the preprocessed data,if false grabs the raw data.
    :param store_metadata: If true, will tell the datapoints reader/writer to save metadata on read.
    :param kwargs: Can overwrite some settings.
    :return: Train and val dataloaders.
    """
    config = get_config_from_dataset(dataset_name)

    train_points, val_points = get_preprocessed_datapoints(dataset_name, get_label_mapping_from_dataset(dataset_name),
                                                           fold) \
        if preprocessed_data else get_raw_datapoints_folded(dataset_name, fold)

    train_dataset = PipelineDataset(train_points, train_transforms, store_metadata=store_metadata)
    val_dataset = PipelineDataset(val_points, val_transforms, store_metadata=store_metadata)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=kwargs.get('batch_size', config['batch_size']),
        num_workers=kwargs.get('processes', config['processes']),
        shuffle=kwargs.get('shuffle', True),
        pin_memory=True,
        collate_fn=batch_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=kwargs.get('batch_size', config['batch_size']),
        num_workers=kwargs.get('processes', config['processes']),
        shuffle=False,
        pin_memory=True,
        collate_fn=batch_collate_fn
    )

    return train_dataloader, val_dataloader
