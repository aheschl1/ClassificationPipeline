import glob
import os
import shutil
from typing import List

from src.utils.constants import PREPROCESSED_ROOT, RAW_ROOT


def maybe_make_preprocessed(dataset_name: str, query_overwrite: bool = True) -> None:
    """
    Checks if the preprocessed folder should be made for a dataset. Makes it if needed.
    :param dataset_name: Dataset name
    :param query_overwrite: Whether to ask before overwriting.
    :return: None
    """
    target_folder = f"{PREPROCESSED_ROOT}/{dataset_name}"
    if os.path.exists(target_folder):
        remove = input(f"Warning! You are about to overwrite the existing path {target_folder}. Continue? (y/n): ") \
                 == "y" if query_overwrite else True
        if not remove:
            print("Killing program.")
            raise SystemExit
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)


def get_labels_from_raw(dataset_name: str) -> List[str]:
    """
    Given a dataset name checks what labels are in the dataset.
    :param dataset_name: Name of the dataset.
    :return: List of labels.
    """
    path = f"{RAW_ROOT}/{dataset_name}"
    folders = glob.glob(f"{path}/*")
    return [f.split('/')[-1] for f in folders]
