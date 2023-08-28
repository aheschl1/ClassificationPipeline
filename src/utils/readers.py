from typing import Type
import SimpleITK as sitk
from constants import *
import numpy as np


class BaseReader:
    def __verify_extension(self, extension: str) -> None:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def read(self, path: str) -> np.array:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")


class SimpleITKReader(BaseReader):
    def __verify_extension(self, extension: str) -> None:
        assert extension is '.gz', f'Invalid extension {extension} for reader SimpleITKReader.'

    def read(self, path: str) -> np.array:
        self.__verify_extension(path.split('.')[-1])
        return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_reader(reader: str) -> Type[BaseReader]:
    assert reader in [SIMPLE_ITK], f'Unrecognized reader {reader}.'
    reader_mapping = {
        SIMPLE_ITK: SimpleITKReader
    }
    return reader_mapping[reader]
