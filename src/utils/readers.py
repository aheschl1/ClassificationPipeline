import SimpleITK as sitk
from constants import *
import numpy as np


class BaseReader:
    def read(self, path: str) -> np.array:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")


class SimpleITKReader(BaseReader):
    def read(self, path: str) -> np.array:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_reader(reader: str):
    assert reader in [SIMPLE_ITK], f'Unrecognized reader {reader}.'
    reader_mapping = {
        SIMPLE_ITK: SimpleITKReader
    }
    return reader_mapping[reader]
