from typing import Union

from classifiers import Classifier, DefaultClassifier
from classifiers.decision_tree_classifier import DecisionTreeClassifier
from classifiers.dnn_classifier import DnnClassifier
from classifiers.random_forest_classifier import RandomForestClassifier


def get_classifier(model: Union[str, None]) -> Classifier:
    if model is None:
        return DefaultClassifier()

    if model == "tree":
        return DecisionTreeClassifier()
    if model == "forest":
        return RandomForestClassifier()
    if model == "dnn":
        return DnnClassifier()

    raise Exception(f"Invalid model: {model}.")
