import torch
import torch.nn as nn
import sys
sys.path.insert(0, "../../..")
from s3ima.arch.ResNet18.model import ResNet18, BasicBlock

GRAYSCALE = True


class ResNet18SimCLR(nn.Module):
    def __init__(self, projection_dim) -> None:
        super(ResNet18SimCLR, self).__init__()

        # set base model as backbone
        self.backbone = ResNet18(num_layers=18,
                                 block=BasicBlock,
                                 num_classes=projection_dim,
                                 grayscale=GRAYSCALE)

        dim_fcl = self.backbone.fc.in_features

        # Add Projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features=dim_fcl, out_features=dim_fcl),
            nn.ReLU(),
            self.backbone.fc
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


