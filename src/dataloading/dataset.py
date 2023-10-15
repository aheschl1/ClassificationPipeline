from typing import List, Callable, Tuple

import torch
from torch.utils.data import Dataset
from src.dataloading.datapoint import Datapoint
from src.utils.constants import SEGMENTATION, CLASSIFICATION


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
        self.dataset_type = datapoints[0].dataset_type

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

    # TODO deal with segmentation augmentation
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Datapoint]:
        """
        Loads the data from the index and transforms it.
        :param idx: The data to grab
        :return: The loaded datapoint
        """

        point = self.datapoints[idx]
        data = point.get_data(store_metadata=self.store_metadata, )
        mask = None
        transform_package = {}
        if self.dataset_type == SEGMENTATION:
            mask = data[1]
            data = data[0]
        # tensor
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        # transforms
        transform_package['image'] = data
        if self.transforms is not None:
            if self.dataset_type == SEGMENTATION:
                transform_package['image'] = transform_package['image'].unsqueeze(0)
                transform_package['mask'] = mask.unsqueeze(0)
            transform_out = self.transforms(transform_package)
            data = transform_out['image']
            if self.dataset_type == SEGMENTATION:
                mask = transform_out['mask'].squeeze(0)
                data = data.squeeze(0)
        point.set_num_classes(self.num_classes)
        # one hot target mask

        # bundle properly for segmentation
        if self.dataset_type == SEGMENTATION:
            data = (data, mask)
        return data, point
