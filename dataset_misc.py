import random
from collections.abc import Iterator
from os import listdir
from typing import Literal

import torch
from PIL.Image import Image, open
from torch.utils.data import IterableDataset
from torchvision.transforms import v2 as transforms


class MVTecLOCOIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: Literal["mvtec_ad", "mvtec_loco"],
        group: Literal["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
        phase: Literal["test", "train", "validation"],
        sorting: str,
    ) -> None:
        self._dataset_name = dataset_name
        self._group = group
        self._phase = phase
        self._sorting = sorting
        self._dir = f"./dataset/{self._dataset_name}/{self._group}/{self._phase}/{self._sorting}/"
        self._files = listdir(self._dir)
        self._len = len(self._files)

    def __iter__(self) -> Iterator[Image]:
        while True:
            yield open(f"{self._dir}{self._files[random.randint(0, self._len - 1)]}")

    def __len__(self) -> int:
        return self._len


class ConvertedHuggingFaceIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterator[Image]:
        while True:
            try:
                yield next(iter(self._dataset))["image"]
            except Exception as err:
                if "Server Error" not in str(err):
                    raise

    def __len__(self) -> int:
        raise NotImplementedError


class TensorConvertedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            image = next(iter(self._dataset))
            image = image.convert("RGB")
            preprocess = transforms.Compose([transforms.ToTensor()])
            yield preprocess(image)

    def __len__(self) -> int:
        return len(self._dataset)


class TransformedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, composed_transforms: transforms.Compose) -> None:
        self._dataset = dataset
        self._composed_transforms = composed_transforms

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            image = next(iter(self._dataset))
            image = self._composed_transforms(image)
            yield image

    def __len__(self) -> int:
        return len(self._dataset)
