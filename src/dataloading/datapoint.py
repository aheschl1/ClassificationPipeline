import numpy as np
from src.utils.constants import *
from src.utils.reader_writer import get_reader, get_writer


class Datapoint:
    def __init__(self, path: str, label: int,
                 case_name: str = None,
                 reader: str = SIMPLE_ITK,
                 writer: str = SIMPLE_ITK) -> None:

        self.path = path
        self.label = label
        self._reader = get_reader(reader)()
        self.writer = get_writer(writer)()
        self._case_name = case_name
        self.data = None

    def get_data(self) -> np.array:
        return self._reader.read(self.path)

    def load_data(self) -> np.array:
        self.data = self._reader.read(self.path)

    @property
    def case_name(self) -> str:
        if self._case_name is not None:
            return self._case_name
        raise NameError('You are trying to access case name when you never set it. set case_name when constructing '
                        'object.')


