import torch

class DiceScoreMetric:
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return dice.item()