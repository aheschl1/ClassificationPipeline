from typing import Type, Union
import SimpleITK as sitk
import torch

from src.utils.constants import *
import numpy as np


class BaseReader:
    def __verify_extension(self, extension: str) -> None:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def read(self, path: str) -> np.array:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")


class SimpleITKReader(BaseReader):
    def __verify_extension(self, extension: str) -> None:
        assert extension == 'nii.gz', f'Invalid extension {extension} for reader SimpleITKReader.'

    def read(self, path: str) -> np.array:
        self.__verify_extension('.'.join(path.split('.')[1:]))
        return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_reader(reader: str) -> Type[BaseReader]:
    assert reader in [SIMPLE_ITK], f'Unrecognized reader {reader}.'
    reader_mapping = {
        SIMPLE_ITK: SimpleITKReader
    }
    return reader_mapping[reader]


class BaseWriter:
    def __verify_extension(self, extension: str) -> None:
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")


class SimpleITKWriter(BaseWriter):
    def __verify_extension(self, extension: str) -> None:
        assert extension == 'nii.gz', f'Invalid extension {extension} for writer SimpleITKWriter.'

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str) -> None:
        self.__verify_extension('.'.join(path.split('.')[1:]))
        if isinstance(data, torch.Tensor):
            data = np.array(data.detach().cpu())
        return sitk.WriteImage(sitk.GetImageFromArray(data), path)


def get_writer(reader: str) -> Type[BaseWriter]:
    assert reader in [SIMPLE_ITK], f'Unrecognized reader {reader}.'
    reader_mapping = {
        SIMPLE_ITK: SimpleITKWriter
    }
    return reader_mapping[reader]
