from os import listdir
from typing import Literal, Optional

import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class MVTecLocoDataset(Dataset):
    def __init__(
        self,
        group: Literal["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
        phase: Literal["test", "train", "validation"],
        output_size: Optional[tuple[int, int]] = None,
    ):
        self._group = group
        self._phase = phase
        self._output_size = output_size
        self._dir = f"./dataset/mvtec_loco/{self._group}/{self._phase}/good/"
        self._files = listdir(self._dir)

    def __getitem__(self, index) -> Image:
        image = read_image(f"{self._dir}{self._files[index]}")
        image = torch.tensor(image, dtype=torch.float32)
        pipeline = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if self._output_size is not None:
            pipeline.append(transforms.Resize(self._output_size, antialias=True))
        preprocess = transforms.Compose(pipeline)
        return preprocess(image)

    def __len__(self):
        return len(self._files)


def InfiniteDataloader(dataloader):
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
