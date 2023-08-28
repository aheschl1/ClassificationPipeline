import numpy as np
from src.utils.constants import *
from src.utils.readers import get_reader


class Datapoint:
    def __init__(self, path: str, label: int, reader: str = SIMPLE_ITK, case_name: str = None) -> None:
        self.path = path
        self.label = label
        self._reader = get_reader(reader)()
        self._case_name = case_name

    def get_data(self) -> np.array:
        return self._reader.read(self.path)

    @property
    def case_name(self) -> str:
        if self._case_name is not None:
            return self._case_name
        raise NameError('You are trying to access case name when you never set it. set case_name when constructing '
                        'object.')


