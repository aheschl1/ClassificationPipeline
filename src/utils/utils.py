import glob
import json
import os
from typing import Dict, List, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataloading.datapoint import Datapoint
from src.dataloading.dataset import PipelineDataset
from src.utils.constants import PREPROCESSED_ROOT, RAW_ROOT, CLASSIFICATION, SEGMENTATION
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm


def write_json(data: Union[Dict, List], path: str, create_folder: bool = False) -> None:
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
    assert len(case_name.split('_')[-1]) >= 5, f"Invalid case name {case_name} in one of your folders. Case name " \
                                               "should be format case_xxxxx, with >= 5 x's"


def get_raw_datapoints(dataset_name: str, label_to_id_mapping: Dict[str, int] = None) -> List[Datapoint]:
    """
    Given the name of a dataset, gets a list of datapoint objects.
    :param dataset_name: The name of the dataset.
    :param label_to_id_mapping: The label to id mapping for converting label name to number.
    :return: List of datapoints in the dataset.
    """
    assert label_to_id_mapping is not None, ("Since this is a classification dataset, expected label_to_id_mapping"
                                             " not to be None.")

    dataset_root = f"{RAW_ROOT}/{dataset_name}"
    datapoints = []
    logging.info("Reading dataset paths.")
    sample_paths = glob.glob(f"{dataset_root}/*/**")
    logging.info("Paths reading has completed.")
    for path in tqdm(sample_paths, desc="Preparing datapoints"):
        # -12 is the label for segmentation to ensure intentionality when building points
        label = label_to_id_mapping[path.split('/')[-2]]
        case_name = path.split('/')[-1].split('.')[0]
        verify_case_name(case_name)
        datapoints.append(Datapoint(path, label, case_name=case_name, dataset_name=dataset_name))

    return datapoints


def get_preprocessed_datapoints(dataset_name: str, fold: int) \
        -> Tuple[List[Datapoint], List[Datapoint]]:
    """
    Returns the datapoints of preprocessed cases.
    :param dataset_name:
    :param fold:
    :return: Train points, Val points.
    """
    train_root = f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/train"
    val_root = f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/val"
    logging.info("Reading dataset paths.")
    val_paths = glob.glob(f"{val_root}/*")
    train_paths = glob.glob(f"{train_root}/*")
    logging.info("Paths reading has completed.")
    sample_paths = val_paths + train_paths
    train_datapoints, val_datapoints = [], []
    label_case_mapping = get_label_case_mapping_from_dataset(dataset_name)
    for path in tqdm(sample_paths, desc="Preparing datapoints"):
        name = path.split('/')[-1].split('.')[0]
        # -12 is sentinel value for segmentation labels to ensure intention.
        label = label_case_mapping[name]
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


def get_config_from_dataset(dataset_name: str, config_name: str = 'config') -> Dict:
    """
    Given a dataset name looks for a config file.
    :param config_name: Name of the config file to load
    :param dataset_name: The name of the dataset.
    :return: Config dictionary.
    """
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/{config_name}.json"
    return read_json(path)


def get_label_case_mapping_from_dataset(dataset_name: str) -> Dict:
    """
    Given a dataset name looks for a case_label_mapping file.
    :param dataset_name: The name of the dataset.
    :return: case_label_mapping dictionary
    """
    path = f"{PREPROCESSED_ROOT}/{dataset_name}/case_label_mapping.json"
    return read_json(path)


def batch_collate_fn(batch: List[Tuple[torch.Tensor, Datapoint]]) -> Tuple[torch.Tensor, torch.Tensor, List[Datapoint]]:
    """
    Combines data fetched by dataloader into proper format.
    :param batch: List of data points from loader.
    :return: Batched tensor data, labels, and list of datapoints.
    """
    data = []
    labels = []
    num_classes = batch[0][1].num_classes
    points = []
    assert num_classes is not None, "All datapoints should have the property " \
                                    "num_classes set before collate_fn. womp womp"
    for data_point, point in batch:
        data.append(data_point)
        points.append(point)
        label = torch.zeros(num_classes)
        label[point.label] = 1
        labels.append(label)

    return torch.stack(data), torch.stack(labels), points


def make_validation_bar_plot(results: Dict[int, int], output_path: str):
    """
    Generates and saves plot of accuracy per class for validation
    """
    quantities, labels = [], []
    for label, quantity in results.items():
        quantities.append(quantity)
        labels.append(label)
    x_axis = np.arange(len(labels))
    plt.bar(x_axis, quantities, 0.4)
    plt.xticks(x_axis, labels)
    plt.xlabel("Label")
    plt.ylabel("Accuracy")
    plt.savefig(f"{output_path}")
    plt.clf()


def get_dataloaders_from_fold(dataset_name: str, fold: int,
                              train_transforms=None,
                              val_transforms=None,
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

    train_points, val_points = get_preprocessed_datapoints(dataset_name, fold) if preprocessed_data \
        else get_raw_datapoints_folded(dataset_name, fold)

    train_dataset = PipelineDataset(train_points, train_transforms,
                                    store_metadata=store_metadata)
    val_dataset = PipelineDataset(val_points, val_transforms,
                                  store_metadata=store_metadata)
    train_sampler, val_sampler = None, None
    if 'sampler' in kwargs and kwargs['sampler'] is not None:
        assert 'rank' in kwargs and 'world_size' in kwargs, \
            "If supplying 'sampler' you must also supply 'world_size' and 'rank'"
        train_sampler = kwargs['sampler'](train_dataset, rank=kwargs['rank'],
                                          num_replicas=kwargs['world_size'], shuffle=True)
        val_sampler = kwargs['sampler'](val_dataset, rank=kwargs['rank'],
                                        num_replicas=kwargs['world_size'], shuffle=False)


    batch_size = kwargs.get('batch_size', config['batch_size'])
    if 'world_size' in kwargs:
        batch_size //= kwargs['world_size']

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['processes'],
        shuffle=train_sampler is None,
        pin_memory=False,
        collate_fn=batch_collate_fn,
        sampler=train_sampler,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config['processes'],
        shuffle=False,
        pin_memory=False,
        collate_fn=batch_collate_fn,
        sampler=val_sampler,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader


def get_case_name_from_number(c: int) -> str:
    """
    Given a case number, returns the string name.
    :param c: The case number.
    :return: The case name in form case_xxxxx
    """
    c = str(c)
    zeros = '0' * (5 - len(c))
    return f"case_{zeros}{c}"


def get_weights_from_dataset(dataset: PipelineDataset) -> List[float]:
    weights = [0 for _ in range(dataset.num_classes)]
    for point in dataset.datapoints:
        weights[point.label] += 1
    total = sum(weights)
    weights = [1. - (weight / total) + 1 for weight in weights]
    return weights
