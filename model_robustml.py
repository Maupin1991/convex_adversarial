import robustml
import torch
import torch.nn as nn

class Model(robustml.model.Model):
  def __init__(self):
    self._model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)).cuda()

    self._model.load_state_dict(torch.load('models/mnist.pth'))

    self._dataset = robustml.dataset.MNIST()
    self._threat_model = robustml.threat_model.Linf(epsilon=0.1)

  @property
  def dataset(self):
      return self._dataset

  @property
  def threat_model(self):
      return self._threat_model

  def classify(self, x):
      X = torch.Tensor([[x]]).cuda()
      out = self._model(X).data.max(1)[1]
      return int(out)


class Flatten(nn.Module):
    def forward(self, x):
            return x.view(x.size(0), -1)
