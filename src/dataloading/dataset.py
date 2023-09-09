from typing import List, Callable

import torch
from torch.utils.data import Dataset

from src.dataloading.datapoint import Datapoint
import psutil


class PipelineDataset(Dataset):

    def __init__(self,
                 datapoints: List[Datapoint],
                 transforms: Callable = None,
                 store_metadata: bool = False
                 ):
        """
        Custom dataset for this pipeline.
        :param datapoints: The list of datapoints for the dataset.
        :param transforms: The transforms to apply to the data.
        :param store_metadata: Whether this data requires metadata storage.
        """
        self.datapoints = datapoints
        self.transforms = transforms
        self.store_metadata = store_metadata
        self.num_classes = self._get_number_of_classes()

    def _get_number_of_classes(self):
        """
        Checks how many classes there are based on classes of datapoints.
        :return: Number of classes in dataset.
        """
        classes = set()
        for point in self.datapoints:
            classes.add(point.label)
        return len(classes)

    def __len__(self):
        """
        Gets the length of dataset.
        :return: Length of datapoints list.
        """
        return len(self.datapoints)

    def __getitem__(self, idx) -> Datapoint:
        """
        Loads the data from the index and transforms it.
        :param idx: The data to grab
        :return: The loaded datapoint
        """
        point = self.datapoints[idx]
        point.load_data(store_metadata=self.store_metadata)
        if not isinstance(point.data, torch.Tensor):
            point.data = torch.Tensor(point.data)
        if self.transforms is not None:
            point.data = self.transforms(point.data)
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
