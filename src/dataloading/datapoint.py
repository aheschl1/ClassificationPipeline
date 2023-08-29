import numpy as np
from src.utils.constants import *
from src.utils.reader_writer import get_reader_writer


class Datapoint:
    def __init__(self, path: str, label: int, dataset_name: str = None,
                 case_name: str = None,
                 writer: str = SIMPLE_ITK) -> None:

        self.path = path
        self.label = label
        self.reader_writer = get_reader_writer(writer)(case_name=case_name, dataset_name=dataset_name)
        self._case_name = case_name
        self.data = None

    def get_data(self, **kwargs) -> np.array:
        return self.reader_writer.read(self.path, **kwargs)

    def load_data(self, **kwargs) -> np.array:
        self.data = self.reader_writer.read(self.path, **kwargs)

    @property
    def case_name(self) -> str:
        if self._case_name is not None:
            return self._case_name
        raise NameError('You are trying to access case name when you never set it. set case_name when constructing '
                        'object.')

    @property
    def extension(self) -> str:
        return '.'.join(self.path.split('.')[1:])


