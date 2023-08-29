from torch.utils.data import Dataset
from typing import List, Callable, Tuple
from src.dataloading.datapoint import Datapoint
import torch


class PipelineDataset(Dataset):
    """
    Custom dataset for this pipeline.
    """
    def __init__(self, datapoints: List[Datapoint], transforms: Callable = None, store_metadata: bool = False):
        self.datapoints = datapoints
        self.transforms = transforms
        self.store_metadata = store_metadata

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx) -> Datapoint:
        point = self.datapoints[idx]
        point.load_data(store_metadata=self.store_metadata)
        if self.transforms is not None:
            point.data = self.transforms(point.data)
        if not isinstance(point.data, torch.Tensor):
            point.data = torch.tensor(point.data)
        return point
