from typing import Tuple, Type, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Normalize
from src.utils.constants import CT, NATURAL


class Normalizer:
    def __init__(self, dataloader: DataLoader, active: bool = True, calculate_early: bool = True) -> None:
        """
        Given a dataloader, will wrap it and perform a normalization operation specific to the type of data.
        :param dataloader: Iterator to wrap
        :param active: If normalization should be applied to the data.
        :param calculate_early: If calculations should be performed. If false, you must sync with another.
        """
        self.active = active
        self.calculate_early = calculate_early
        self.mean, self.std = None, None
        self.length = len(dataloader)
        self._init(dataloader)
        self.dataloader = iter(dataloader)

    def _init(self, dataloader: DataLoader) -> None:
        """
        Performs calculations on the data.
        :param dataloader: The wrapper dataloader.
        :return: Nothing
        """
        raise NotImplemented('Do not use the base class as an iterator.')

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Next step in the iterator. If not active, just mirror the original dataloader.
        :return:
        """
        if not self.active:
            return next(self.dataloader)
        return self._normalize(*next(self.dataloader))

    def sync(self, other):
        """
        Syncs one normalizer with another. This is to share calculations across similar sets.
        :param other: The other normalizer to sync with.
        :return:
        """
        assert self.__class__ == other.__class__, \
            f"You tried syncing type {self.__class__} with type {other.__class__}."

    def _normalize(self,
                   data: torch.Tensor,
                   label: torch.Tensor,
                   point: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Method where data normalization is computed.
        :param data: The data to normalize.
        :param label:
        :param point:
        :return: Normalized data and other two points
        """
        pass


class NaturalImageNormalizer(Normalizer):
    def _init(self, dataloader: DataLoader):
        """
        Computes three channel mean and std for natural image normalization.
        :param dataloader:
        :return:
        """
        if not self.active or not self.calculate_early:
            return
        means = []
        for data, _, _ in tqdm(dataloader, desc="Calculating mean"):
            assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
            assert data.shape[3] == 3, f"NaturalImageNormalizer requires three channels, and shape [b, h, w, c]." \
                                       f"Got {data.shape}"
            means.append(torch.mean(data.float(), dim=[0, 1, 2]))

        means = torch.stack(means)
        mu_rgb = torch.mean(means, dim=[0])
        variances = []
        for data, _, _ in tqdm(dataloader, desc="Calculating std"):
            assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
            var = torch.mean((data - mu_rgb) ** 2, dim=[0, 1, 2])
            variances.append(var)
        variances = torch.stack(variances)
        std_rgb = torch.sqrt(torch.mean(variances, dim=[0]))
        self.mean = mu_rgb
        self.std = std_rgb

    def sync(self, other):
        super().sync(other)
        self.mean = other.mean
        self.std = other.std

    def _normalize(self,
                   data: torch.Tensor,
                   label: torch.Tensor,
                   point: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        return Normalize(mean=self.mean, std=self.std)(data.float().permute(0, 3, 1, 2)), label, point


class CTNormalizer(Normalizer):
    def _init(self, dataloader: DataLoader) -> None:
        """
        Computes mean and std across entire image for CT normalization.
        :param dataloader:
        :return:
        """
        if not self.active or not self.calculate_early:
            return
        psum = torch.tensor([0.0])
        psum_sq = torch.tensor([0.0])
        pixel_count = 0

        # loop through images
        for data, _, _ in tqdm(dataloader, desc="Calculating mean and std"):
            assert data.shape[0] == 1, "Expected batch size 1 for mean std calculations. Womp womp"
            psum += data.sum()
            psum_sq += (data ** 2).sum()

            pixels = 1.
            for i in data.shape:
                pixels *= i
            pixel_count += pixels

        # mean and std
        total_mean = psum / pixel_count
        total_var = (psum_sq / pixel_count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)

        # output
        self.mean = total_mean
        self.std = total_std

    def sync(self, other):
        super().sync(other)
        self.mean = other.mean
        self.std = other.std

    def _normalize(self,
                   data: torch.Tensor,
                   label: torch.Tensor,
                   point: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        return Normalize(mean=self.mean, std=self.std)(data.float().permute(0, 3, 1, 2)), label, point


def get_normalizer(norm: str) -> Type[Normalizer]:
    assert norm in [NATURAL, CT], f'Unrecognized normalizer type {norm}.'
    norm_mapping = {
        CT: CTNormalizer,
        NATURAL: NaturalImageNormalizer
    }
    return norm_mapping[norm]


def get_normalizer_from_extension(extension: str) -> Type[Normalizer]:
    mapping = {
        'nii.gz': CTNormalizer,
        'png': NaturalImageNormalizer,
        'jpg': NaturalImageNormalizer,
        'npy': NaturalImageNormalizer
    }
    assert extension in mapping.keys(), f"Currently unsupported extension {extension}"
    return mapping[extension]
