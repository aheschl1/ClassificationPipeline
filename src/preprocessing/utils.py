from src.utils.constants import PREPROCESSED_ROOT, RAW_ROOT
from src.utils.utils import verify_case_name
import shutil
import os
from typing import List, Dict, Tuple
from src.dataloading.datapoint import Datapoint
import glob
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


def maybe_make_preprocessed(dataset: str, query_overwrite: bool = True) -> None:
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
    path = f"{RAW_ROOT}/{dataset_name}"
    folders = glob.glob(f"{path}/*")
    return [f.split('/')[-1] for f in folders]


def calculate_mean_std(dataloader: DataLoader) -> Tuple[float, float]:
    """
    Returns mean and std of entire dataset, not calculated across channel
    :param dataloader: Desired dataloader
    :return: mean and std
    """
    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])
    pixel_count = 0

    # loop through images
    for data, _ in tqdm(dataloader, desc="Calculating mean and std"):
        assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
        psum += data.sum()
        psum_sq += (data ** 2).sum()
        pixels = 1.
        for i in data.shape[1:]:
            pixels *= i
        pixel_count += pixels

    # mean and std
    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    return total_mean.item(), total_std.item()
