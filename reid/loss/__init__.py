from __future__ import absolute_import

from .triplet import TripletLoss
from .lsr import LSRLoss
from .capsuleloss import CapsuleLoss

__all__ = [
    'TripletLoss',
    'LSRLoss',
    'CapsuleLoss',
]
