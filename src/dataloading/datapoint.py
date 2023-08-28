import numpy as np
from ..utils.constants import *


class Datapoint:
    def __init__(self, path: str, label: int) -> None:
        self.path = path
        self.label = label

    def get_data(self, reader: str = SIMPLE_ITK) -> np.array:
        assert reader in [SIMPLE_ITK], 'Unsupported reader specified'

