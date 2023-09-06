from typing import List, Callable

import torch
from torch.utils.data import Dataset

from src.dataloading.datapoint import Datapoint
import psutil


class PipelineDataset(Dataset):
    """
    Custom dataset for this pipeline.
    """
    def __init__(self,
                 datapoints: List[Datapoint],
                 transforms: Callable = None,
                 store_metadata: bool = False
                 ):
        self.datapoints = datapoints
        self.transforms = transforms
        self.store_metadata = store_metadata
        self.num_classes = self._get_number_of_classes()
        self.preloaded_data = {}

    def _get_number_of_classes(self):
        classes = set()
        for point in self.datapoints:
            classes.add(point.label)
        return len(classes)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx) -> Datapoint:
        point = self.datapoints[idx]
        if point.path not in self.preloaded_data:
            point.load_data(store_metadata=self.store_metadata)
        else:
            point.data = self.preloaded_data.pop(point.path)
        if self.transforms is not None:
            point.data = self.transforms(point.data)
        if not isinstance(point.data, torch.Tensor):
            point.data = torch.tensor(point.data)
        point.set_num_classes(self.num_classes)
        return point

    @staticmethod
    def available_memory() -> float:
        """
        Method for getting available memory.
        :return: Total available memory in bytes.
        """
        return psutil.virtual_memory()[1]


if __name__ == "__main__":
    print(PipelineDataset.available_memory())
