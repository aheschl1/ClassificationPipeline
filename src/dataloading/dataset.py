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

    def __getitem__(self, idx) -> Tuple[torch.tensor, int]:
        point = self.datapoints[idx]
        data = point.get_data(store_metadata=self.store_metadata)
        if self.transforms is not None:
            data = self.transforms(data)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        return data, point.label
