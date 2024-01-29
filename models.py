from torch import nn

# return patch description model (PDN)
def get_pdn(channels: int = 512):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=channels, kernel_size=4, stride=1, padding=0)
    )