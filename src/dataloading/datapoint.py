import numpy as np

from src.utils.normalizer import get_normalizer, get_normalizer_from_extension
from src.utils.reader_writer import get_reader_writer, get_reader_writer_from_extension


class Datapoint:
    def __init__(self, path: str, label: int, dataset_name: str = None,
                 case_name: str = None,
                 writer: str = None,
                 normalizer: str = None) -> None:

        self.path = path
        self.label = label
        # reader
        if writer is not None:
            self.reader_writer = get_reader_writer(writer)
        else:
            self.reader_writer = get_reader_writer_from_extension(self.extension)
        # normalizer
        if normalizer is not None:
            self.normalizer = get_normalizer(writer)
        else:
            self.normalizer = get_normalizer_from_extension(self.extension)
        self.reader_writer = self.reader_writer(case_name=case_name, dataset_name=dataset_name)
        self._case_name = case_name
        self.num_classes = None

    def get_data(self, **kwargs) -> np.array:
        return self.reader_writer.read(self.path, **kwargs)

    def set_num_classes(self, n: int) -> None:
        self.num_classes = n

    @property
    def case_name(self) -> str:
        if self._case_name is not None:
            return self._case_name
        raise NameError('You are trying to access case name when you never set it. set case_name when constructing '
                        'object.')

    @property
    def extension(self) -> str:
        name = self.path.split('/')[-1]
        return '.'.join(name.split('.')[1:])
