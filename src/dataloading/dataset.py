from typing import List, Callable, Tuple

import torch
from torch.utils.data import Dataset
from src.dataloading.datapoint import Datapoint


class PipelineDataset(Dataset):

    def __init__(self,
                 datapoints: List[Datapoint],
                 transforms: Callable = None,
                 store_metadata: bool = False,
                 preload: bool = False,
                 dataset_type: str = "train"
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
        self._num_classes = None
        self.num_classes = self._get_number_of_classes()
        self.preload = preload
        self.dataset_type = dataset_type

    def _get_number_of_classes(self):
        """
        Checks how many classes there are based on classes of datapoints.
        :return: Number of classes in dataset.
        """
        if self._num_classes is not None:
            return self._num_classes
        classes = set()
        for point in self.datapoints:
            classes.add(point.label)
        self._num_classes = len(classes)
        return self._num_classes
    
    def __len__(self):
        """
        Gets the length of dataset.
        :return: Length of datapoints list.
        """
        return len(self.datapoints)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Datapoint]:
        """
        Loads the data from the index and transforms it.
        :param idx: The data to grab
        :return: The loaded datapoint
        """

        point = self.datapoints[idx]
        data = point.get_data(store_metadata=self.store_metadata, )

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        if self.transforms is not None:
            data = self.transforms(data)
        point.set_num_classes(self.num_classes)
        return data, point
