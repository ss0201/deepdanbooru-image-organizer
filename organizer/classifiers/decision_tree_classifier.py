import pickle

import numpy as np
from sklearn import tree
from util.frequent_safety_tags import FREQUENT_SAFETY_TAGS
from util.print_buffer import PrintBuffer

from classifiers.classifier import Classifier


class DecisionTreeClassifier(Classifier):
    model: tree.DecisionTreeClassifier

    def __init__(self, model_path: str) -> None:
        super().__init__()
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def get_classification(
        self, evaluation_dict: dict[str, int], print_buffer: PrintBuffer
    ) -> str:
        evaluations = [evaluation_dict[x] for x in FREQUENT_SAFETY_TAGS]
        evaluations = np.reshape(evaluations, (1, -1))
        return self.model.predict(evaluations)[0]
