import glob
import os
import shutil
from typing import List, Tuple, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.constants import PREPROCESSED_ROOT, RAW_ROOT


def maybe_make_preprocessed(dataset: str, query_overwrite: bool = True) -> None:
    """
    Checks if the preprocessed folder should be made for a dataset. Makes it if needed.
    :param dataset: Dataset name
    :param query_overwrite: Whether to ask before overwriting.
    :return: None
    """
    target_folder = f"{PREPROCESSED_ROOT}/{dataset}"
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


def _calculate_natural_image_mean_std(dataloader: DataLoader) -> Tuple[Any, Any]:
    means = []
    for data, _, _ in tqdm(dataloader, desc="Calculating mean"):
        assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
        means.append(torch.mean(data.float(), dim=[0, 1, 2]))

    means = torch.stack(means)
    mu_rgb = torch.mean(means, dim=[0])
    variances = []
    for data, _, _ in tqdm(dataloader, desc="Calculating std"):
        assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
        var = torch.mean((data - mu_rgb) ** 2, dim=[0, 1, 2])
        variances.append(var)
    variances = torch.stack(variances)
    std_rgb = torch.sqrt(torch.mean(variances, dim=[0]))
    return mu_rgb, std_rgb


def calculate_mean_std(dataloader: DataLoader) -> Tuple[Any, Any]:
    """
    Returns mean and std of entire dataset, calculated across channel for 3 channel inputs.
    Expects [b, w, h, c]
    :param dataloader: Desired dataloader
    :return: mean and std
    """
    extension = dataloader.dataset[0].extension
    if extension in ['jpg', 'png']:
        return _calculate_natural_image_mean_std(dataloader)

    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])
    pixel_count = 0

    # loop through images
    for data, _, _ in tqdm(dataloader, desc="Calculating mean and std"):
        assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
        psum += data.sum()
        psum_sq += (data ** 2).sum()

        pixels = 1.
        for i in data.shape:
            pixels *= i
        pixel_count += pixels

    # mean and std
    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    return total_mean, total_std
