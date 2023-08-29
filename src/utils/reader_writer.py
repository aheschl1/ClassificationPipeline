import os
from typing import Type, Union
import SimpleITK as sitk
import torch

from src.utils.constants import *
import numpy as np
import pickle


class BaseReaderWriter:
    def __init__(self, **kwargs):
        pass

    def __verify_extension(self, extension: str) -> None:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def read(self, path: str, **kwargs) -> np.array:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")


class SimpleITKReaderWriter(BaseReaderWriter):

    def __init__(self, case_name: str, dataset_name: str = None):
        super().__init__()
        self.direction = None
        self.spacing = None
        self.origin = None
        self.has_read = False
        self.case_name = case_name
        self.dataset_name = dataset_name

    def __verify_extension(self, extension: str) -> None:
        assert extension == 'nii.gz', f'Invalid extension {extension} for reader SimpleITKReader.'

    def __store_metadata(self) -> None:
        assert self.dataset_name is not None, "Can not store metadata from SimpleITK reader/writer without knowing " \
                                              "dataset name."
        expected_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}/metadata"
        expected_file = f"{expected_folder}/{self.case_name}.pkl"
        data = {
            'spacing': self.spacing,
            'direction': self.direction,
            'origin': self.origin
        }
        if os.path.exists(expected_file):
            os.remove(expected_file)
        if not os.path.exists(expected_folder):
            os.makedirs(expected_folder)
        with open(expected_file, 'wb') as file:
            return pickle.dump(data, file)

    def read(self, path: str, store_metadata: bool = False) -> np.array:
        self.has_read = True
        self.__verify_extension('.'.join(path.split('.')[1:]))
        image = sitk.ReadImage(path)
        self.spacing = image.GetSpacing()
        self.direction = image.GetDirection()
        self.origin = image.GetOrigin()
        if store_metadata:
            self.__store_metadata()
        return sitk.GetArrayFromImage(image)

    def check_for_metadata_folder(self) -> Union[dict, None]:
        expected_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}/metadata"
        expected_file = f"{expected_folder}/{self.case_name}.pkl"
        if os.path.exists(expected_file):
            with open(expected_file, 'rb') as file:
                return pickle.load(file)
        return None

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        if not self.has_read:
            meta = self.check_for_metadata_folder()
            if meta is None:
                raise ValueError(f'SimpleITK reader writer can not find metadata for this image {self.case_name}. If '
                                 f'you read first we can save.')
            try:
                self.spacing = meta['spacing']
                self.direction = meta['direction']
                self.origin = meta['origin']
            except KeyError:
                raise ValueError(f'Invalid metadata found for {self.case_name} in SimpleITKReaderWriter.')
        self.__verify_extension('.'.join(path.split('.')[1:]))
        if isinstance(data, torch.Tensor):
            data = np.array(data.detach().cpu())
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        return sitk.WriteImage(image, path)


def get_reader_writer(io: str) -> Type[BaseReaderWriter]:
    assert io in [SIMPLE_ITK], f'Unrecognized reader/writer {io}.'
    reader_writer_mapping = {
        SIMPLE_ITK: SimpleITKReaderWriter
    }
    return reader_writer_mapping[io]
