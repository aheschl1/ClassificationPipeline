import numpy as np
from ..utils.constants import *
from ..utils.readers import get_reader


class Datapoint:
    def __init__(self, path: str, label: int, reader: str = SIMPLE_ITK) -> None:
        self.path = path
        self.label = label
        self._reader = get_reader(reader)()

    def get_data(self) -> np.array:
        return self._reader.read(self.path)


