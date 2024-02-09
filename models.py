import timm
import torch
from torch import nn


class ChoppedWideResNet(nn.Module):
    def __init__(self, channels: int, layer_to_extract_from: str, layer_index: int = -1) -> None:
        super().__init__()
        self._backbone = timm.create_model("wide_resnet101_2", pretrained=True)
        self._output = None
        self._channels = channels

        def forward_hook(m, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
            self._output = outputs
            raise self.LayerReachedException

        for layer in self._backbone.named_children():
            if layer[0] == layer_to_extract_from:
                layer[1][layer_index].register_forward_hook(forward_hook)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                self._backbone.forward(input)
            except self.LayerReachedException:
                pass
        return self._output[:, : self._channels, :, :]

    class LayerReachedException(Exception):
        pass


class PatchDescriptionNetwork(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=3)
        self.avgpool_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=3)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=self.channels, kernel_size=4, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.avgpool_1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv4(x)
        return x


class NormalizedPatchDescriptionNetwork(nn.Module):
    def __init__(self, pdn: PatchDescriptionNetwork) -> None:
        super().__init__()

        self.pdn = pdn
        self.normalization = nn.BatchNorm2d(num_features=self.pdn.channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pdn(x)
        x = self.normalization(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        # encoding
        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.enc_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.enc_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=0)

        # decoding
        self.dec_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.dec_dropout1 = nn.Dropout(p=0.2)
        self.dec_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.dec_dropout2 = nn.Dropout(p=0.2)
        self.dec_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.dec_dropout3 = nn.Dropout(p=0.2)
        self.dec_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.dec_dropout4 = nn.Dropout(p=0.2)
        self.dec_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.dec_dropout5 = nn.Dropout(p=0.2)
        self.dec_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.dec_dropout6 = nn.Dropout(p=0.2)
        self.dec_conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dec_conv8 = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x) -> torch.Tensor:
        # encoding
        x = self.enc_conv1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.enc_conv2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.enc_conv3(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.enc_conv4(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.enc_conv5(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.enc_conv6(x)

        # decoding
        x = nn.functional.interpolate(x, size=3, mode="bilinear")
        x = self.dec_conv1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_dropout1(x)
        x = nn.functional.interpolate(x, size=8, mode="bilinear")
        x = self.dec_conv2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_dropout2(x)
        x = nn.functional.interpolate(x, size=15, mode="bilinear")
        x = self.dec_conv3(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_dropout3(x)
        x = nn.functional.interpolate(x, size=32, mode="bilinear")
        x = self.dec_conv4(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_dropout4(x)
        x = nn.functional.interpolate(x, size=63, mode="bilinear")
        x = self.dec_conv5(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_dropout5(x)
        x = nn.functional.interpolate(x, size=127, mode="bilinear")
        x = self.dec_conv6(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_dropout6(x)
        x = nn.functional.interpolate(x, size=64, mode="bilinear")
        x = self.dec_conv7(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.dec_conv8(x)

        return x
