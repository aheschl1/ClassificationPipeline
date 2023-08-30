from typing import List, Callable

import torch
from torch.utils.data import Dataset

from src.dataloading.datapoint import Datapoint


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

    def _get_number_of_classes(self):
        classes = set()
        for point in self.datapoints:
            classes.add(point.label)
        return len(classes)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx) -> Datapoint:
        point = self.datapoints[idx]
        point.load_data(store_metadata=self.store_metadata)
        if self.transforms is not None:
            point.data = self.transforms(point.data)
        if not isinstance(point.data, torch.Tensor):
            point.data = torch.tensor(point.data)
        point.set_num_classes(self.num_classes)
        return point
