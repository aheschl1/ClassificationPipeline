import shutil

from src.utils.utils import write_json, get_dataset_name_from_id, check_raw_exists, get_raw_datapoints, \
    get_dataloaders_from_fold
from src.preprocessing.utils import maybe_make_preprocessed, get_labels_from_raw, calculate_mean_std
from src.dataloading.datapoint import Datapoint
from src.utils.constants import *
from src.preprocessing.splitting import Splitter
import click
from typing import Dict, List, Union, Tuple
from tqdm import tqdm
import torch
from torchvision.transforms import Normalize
import time

def build_config(dataset_name: str, processes: int) -> None:
    """
    Creates the config.json file that should contain training hyperparameters. Hardcode default values here.
    :param dataset_name: Name of dataset.
    :param processes: Worker count.
    :return: None
    """
    config = {
        'batch_size': 8,
        'processes': processes
    }
    write_json(config, f"{PREPROCESSED_ROOT}/{dataset_name}/config.json")


def get_folds(k: int, data: List[Datapoint]) -> Dict[int, Dict[str, list]]:
    """
    Gets random fold at 80/20 split. Returns in a map.
    :param k: How many folds for kfold cross validation.
    :param data: List of datapoints. Datapoint objects must have case_name set.
    :return: Folds map
    """
    splitter = Splitter(data, k)
    return splitter.get_split_map()


def map_labels_to_id(labels: List[str], return_inverse: bool = False) -> Union[
    Tuple[Dict[str, int], Dict[int, str]], Dict[str, int]]:
    """
    :param labels: List of string label names.
    :param return_inverse: If true returns id:name as well as name:id mapping.
    :return: Dict that maps label name to id.
    """
    mapping = {}
    inverse = {}

    for i, label in enumerate(labels):
        mapping[label] = i
        inverse[i] = label

    if return_inverse:
        return mapping, inverse
    return mapping


def get_case_to_label_mapping(datapoints: List[Datapoint]) -> Dict[str, int]:
    """
    Given a list of datapoints, we create a mapping of label name to label id.
    :param datapoints:
    :return:
    """
    mapping = {}
    for point in datapoints:
        mapping[point.case_name] = point.label
    return mapping


def process_fold(dataset_name: str, fold: int):
    print(f"Now starting with fold {fold}...")
    time.sleep(1)
    train_loader, val_loader = get_dataloaders_from_fold(dataset_name, fold,
                                                         preprocessed_data=False,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         store_metadata=True
                                                         )
    # prep dirs
    os.mkdir(f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}")
    os.mkdir(f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/train")
    os.mkdir(f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/val")
    # start saving preprocessed stuff
    mean, std = calculate_mean_std(train_loader)
    for _set in ['train', 'val']:
        for data, labels, points in \
                tqdm(train_loader if _set == 'train' else val_loader, desc=f"Preprocessing {_set} set"):
            point = points[0]
            writer = point.reader_writer
            writer.write(
                Normalize(mean=mean, std=std)(data[0].float().squeeze()),
                f"{PREPROCESSED_ROOT}/{dataset_name}/fold_{fold}/{_set}/{point.case_name}.{point.extension}"
            )


@click.command()
@click.option('--folds', '-f', help='How many folds should be generated.', type=int)
@click.option('--processes', '-p', help='How many processes can be used.', type=int, default=8)
@click.option('--normalize', '--n', help='Should we compute and save normalized data.', type=bool,
              default=True, is_flag=True)
@click.option('--dataset_id', '-d', help='The dataset id to work on.', type=str)
def main(folds: int, processes: int, normalize: bool, dataset_id: str):
    """
    :param folds: How many folds to generate.
    :param processes: How many processes should be used.
    :param normalize: Should normalized data be saved.
    :param dataset_id: The id of the dataset.

    This is the main driver for preprocessing.
    """
    dataset_name = get_dataset_name_from_id(dataset_id)
    assert check_raw_exists(dataset_name), f"It appears that you haven't created the 'raw/{dataset_name}' folder. " \
                                           f"Womp womp"
    maybe_make_preprocessed(dataset_name, query_overwrite=False)
    # We export the config building to a new method
    build_config(dataset_name, processes)
    # Here we will find what labels are present in the dataset. We will also map them to int labels, and save the
    # mappings.
    labels = get_labels_from_raw(dataset_name)
    label_to_id_mapping, id_to_label_mapping = map_labels_to_id(labels, return_inverse=True)
    write_json(label_to_id_mapping, f"{PREPROCESSED_ROOT}/{dataset_name}/label_to_id.json")
    write_json(id_to_label_mapping, f"{PREPROCESSED_ROOT}/{dataset_name}/id_to_label.json")
    # Label stuff done, start with fetching data. We will also save a case to label mapping.
    datapoints = get_raw_datapoints(dataset_name, label_to_id_mapping)
    case_to_label_mapping = get_case_to_label_mapping(datapoints)
    write_json(case_to_label_mapping, f"{PREPROCESSED_ROOT}/{dataset_name}/case_label_mapping.json")
    splits_map = get_folds(folds, datapoints)
    write_json(splits_map, f"{PREPROCESSED_ROOT}/{dataset_name}/folds.json")
    # We now have the folds: time to preprocess the data
    for fold_id, _ in splits_map.items():
        process_fold(dataset_name, fold_id)

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
