from multiprocessing.shared_memory import SharedMemory

import numpy as np

from src.utils.normalizer import get_normalizer, get_normalizer_from_extension
from src.utils.reader_writer import get_reader_writer, get_reader_writer_from_extension


class Datapoint:
    def __init__(self, path: str, label: int, dataset_name: str = None,
                 case_name: str = None,
                 writer: str = None,
                 normalizer: str = None,
                 cache: bool = False) -> None:
        """
        Datapoint object. Supports caching, and determines own read/write and normalizer.
        :param path: Path to image on disk
        :param label: Label of image
        :param dataset_name: Name of the dataset. ex: Dataset_000
        :param case_name: The name of the case. ex: case_00000
        :param writer: Overwrite writer class
        :param normalizer: Overwrite normalizer
        :param cache: If enabled, stores data in shared memory
        """

        self.path = path
        self.label = label
        self.cache = cache
        self._shared_mem = None
        self._shape = None
        self._dtype = None
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
        """
        Returns datapoint data. Checks cache if enabled.
        :param kwargs:
        :return: Data as np.array. DO NOT EDIT DIRECTLY IF CACHE ENABLED!!!
        """
        if not self.cache:
            return self.reader_writer.read(self.path, **kwargs)
        if self._shared_mem is None:
            data = self.reader_writer.read(self.path, **kwargs)
            self._shape = data.shape
            self._dtype = data.dtype
            self._cache(data)
            return data
        return self._get_cached()

    def _cache(self, data: np.array) -> None:
        """
        Cache the data
        :return: None
        """
        try:
            self._shared_mem = SharedMemory(size=data.nbytes, create=True, name=self._case_name)
            temp_buf = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shared_mem.buf)
            temp_buf[:] = data
        except FileExistsError: ...  # Already exists

    def _get_cached(self) -> np.array:
        """
        Fetch the cached data
        :return: The data
        """
        return np.asarray(np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared_mem.buf))

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

    def __del__(self):
        """
        Unlink shared memory
        :return:
        """
        if self._shared_mem is not None:
            self._shared_mem.unlink()