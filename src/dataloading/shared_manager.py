from multiprocessing import shared_memory
from typing import Type

import numpy as np


class SharedMemoryManager:

    @staticmethod
    def store_array(array: Type[np.array], idx: int, set_type: str) -> None:
        buff = shared_memory.SharedMemory(create=True, name=f"{idx}_{set_type}", size=array.shape*array.itemsize)
        temp = np.ndarray(array.shape, dtype=array.dtype, buffer=buff.buf)
        np.copyto(temp, array)
        buff.close()

    @staticmethod
    def get_array(idx: int, set_type: str, expected_shape: tuple) -> np.array:
        return np.ndarray(expected_shape, buffer=shared_memory.SharedMemory(name=f"{idx}_{set_type}").buf)

