import torch
from torch import nn
import torch.nn.functional as F

# class CapsuleLoss(nn.Module):
#     def __init__(self):
#         super(CapsuleLoss, self).__init__()
#         self.reconstruction_loss = nn.MSELoss(size_average=False)
#
#     def forward(self, images, labels, classes, reconstructions):
#         left = F.relu(0.9 - classes, inplace=True) ** 2
#         right = F.relu(classes - 0.1, inplace=True) ** 2
#
#         margin_loss = labels * left + 0.5 * (1. - labels) * right
#         margin_loss = margin_loss.sum()
#
#         assert torch.numel(images) == torch.numel(reconstructions)
#         images = images.view(reconstructions.size()[0], -1)
#         reconstruction_loss = self.reconstruction_loss(reconstructions, images)
#
#         return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, images, labels, results):
        left = F.relu(0.9987 - results, inplace=True) ** 2
        right = F.relu(results - 0.0013, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()


        return margin_loss / images.size(0)