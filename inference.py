from typing import Optional, Union

import torch
import yaml
from PIL.Image import Image
from torchvision import transforms


class EfficientADInferencer(torch.nn.Module):
    def __init__(self, model_config_path: str = "model_config.yaml", device: Optional[str] = None) -> None:
        super().__init__()
        self._device = device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(model_config_path) as config_file:
            self._model_config = yaml.safe_load(config_file)

        self._teacher_pdn = torch.load(self._model_config["teacher_path"], map_location=self._device)
        self._autoencoder = torch.load(self._model_config["autoencoder_path"], map_location=self._device)
        self._student_pdn = torch.load(self._model_config["student_path"], map_location=self._device)

    def forward(self, image: Union[torch.Tensor, Image]) -> torch.Tensor:
        with torch.no_grad():
            input = self.preprocess(image)
            teacher_result = self._teacher_pdn.forward(input)
            # print(teacher_result.shape)
            student_result = self._student_pdn.forward(input)
            # print(student_result.shape)
            autoencoder_result = self._autoencoder.forward(input)
            # print(autoencoder_result.shape)

            local_anomalies_map, _ = torch.max(
                (teacher_result - student_result[:, : self._model_config["out_channels"]["teacher"], :, :]) ** 2, dim=1
            )
            global_anomalies_map, _ = torch.max(
                (autoencoder_result - student_result[:, self._model_config["out_channels"]["teacher"] :, :, :]) ** 2,
                dim=1,
            )

            anomaly_map = local_anomalies_map + global_anomalies_map
        return anomaly_map

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
