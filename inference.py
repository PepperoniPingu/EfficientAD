from typing import Optional, Union

import torch
import yaml
from PIL.Image import Image
from torchvision.transforms import v2 as transforms


class EfficientADInferencer(torch.nn.Module):
    def __init__(
        self, model_config_path: str = "model_config.yaml", device: Optional[torch.DeviceObjType] = None
    ) -> None:
        super().__init__()
        self._device = device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(model_config_path) as config_file:
            self._model_config = yaml.safe_load(config_file)

        self._teacher_pdn = torch.load(self._model_config["teacher_path"], map_location=self._device)
        self._autoencoder = torch.load(self._model_config["autoencoder_path"], map_location=self._device)
        self._student_pdn = torch.load(self._model_config["student_path"], map_location=self._device)
        self._quantiles = torch.load(self._model_config["quantiles_path"])  # , map_location=self._device)

    @torch.no_grad()
    def forward(self, image: Union[torch.Tensor, Image]) -> torch.Tensor:
        input = self.preprocess(image)
        teacher_result = self._teacher_pdn.forward(input)
        student_result = self._student_pdn.forward(input)
        autoencoder_result = self._autoencoder.forward(input)

        student_anomaly_map = torch.mean(
            (teacher_result - student_result[:, : self._model_config["out_channels"]["teacher"], :, :]) ** 2, dim=1
        )
        autoencoder_anomaly_map = torch.mean(
            (autoencoder_result - student_result[:, self._model_config["out_channels"]["teacher"] :, :, :]) ** 2,
            dim=1,
        )

        student_anomaly_map = (
            0.1
            * (student_anomaly_map - self._quantiles["student_a"])
            / (self._quantiles["student_b"] - self._quantiles["student_a"])
        )
        autoencoder_anomaly_map = (
            0.1
            * (autoencoder_anomaly_map - self._quantiles["autoencoder_a"])
            / (self._quantiles["autoencoder_b"] - self._quantiles["autoencoder_a"])
        )

        anomaly_map = (student_anomaly_map + autoencoder_anomaly_map) / 2
        anomaly_map = torch.clamp(anomaly_map, 0, 1)
        score = torch.max(anomaly_map)
        return anomaly_map, score

    def preprocess(self, image: Union[torch.Tensor, Image]) -> torch.Tensor:
        input = torch.as_tensor(image, device=self._device)
        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        res = preprocess(input).unsqueeze(0)
        return res

    def create_heat_map(self, image: Union[torch.Tensor, Image], anomaly_map: torch.Tensor) -> torch.Tensor:
        image = torch.as_tensor(image, device=self._device)
        image = image * 0.5
        anomaly_map = transforms.functional.resize(anomaly_map, size=(image.shape[1], image.shape[2]))
        anomaly_map_r = anomaly_map
        anomaly_map_g = anomaly_map * 0
        anomaly_map_b = anomaly_map * 0
        anomaly_map = torch.cat([anomaly_map_r, anomaly_map_g, anomaly_map_b], dim=0)
        heat_map = (image + anomaly_map) / 2
        return heat_map
