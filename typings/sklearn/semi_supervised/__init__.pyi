"""
This type stub file was generated by pyright.
"""

from ._label_propagation import LabelPropagation, LabelSpreading
from ._self_training import SelfTrainingClassifier

"""
The :mod:`sklearn.semi_supervised` module implements semi-supervised learning
algorithms. These algorithms utilize small amounts of labeled data and large
amounts of unlabeled data for classification tasks. This module includes Label
Propagation.
"""
__all__ = ["SelfTrainingClassifier", "LabelPropagation", "LabelSpreading"]
