import argparse

import torch
import torchshow

from dataset_misc import MVTecLOCOIterableDataset, TensorConvertedIterableDataset
from inference import EfficientADInferencer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    efficientad = EfficientADInferencer(device=device)

    dataset = TensorConvertedIterableDataset(
        MVTecLOCOIterableDataset(dataset_name="mvtec_loco", group="splicing_connectors", phase="test", sorting="good")
    )

    for image in dataset:
        anomaly_map, score = efficientad.forward(image)
        print(score)
        heat_map = efficientad.create_heat_map(image=image, anomaly_map=anomaly_map)
        torchshow.show(heat_map)


if __name__ == "__main__":
    main()
