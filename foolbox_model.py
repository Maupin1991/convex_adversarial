import foolbox
import torch
from torch import nn


def create():
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10))

    weights_path = 'models/mnist.pth'
    weights = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights)

    fmodel = foolbox.models.PyTorchModel(model, (0, 1), num_classes=10)
    return fmodel
