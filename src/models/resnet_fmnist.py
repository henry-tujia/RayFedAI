from src.models.Resnet_ import resnet8
import torch


class resnet8_fmnist(torch.nn.Module):
    def __init__(self, num_classes, KD=False, projection=False) -> None:
        super().__init__()
        self.model = resnet8(num_classes, KD=KD, projection=projection)

        self.model.conv_1 = torch.nn.Conv2d(
            1, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.avgpool = torch.nn.Identity()
        self.model.fc = torch.nn.Linear(1024, num_classes)
        if projection:
            self.model.projection_layer = torch.nn.Sequential(
                torch.nn.Linear(1024, 1024),
                torch.nn.Linear(1024, 1024),
            )

    def forward(self, x):
        return self.model(x)
