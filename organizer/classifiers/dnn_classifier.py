import numpy as np
import tensorflow as tf
from tensorflow import keras
from util.frequent_safety_tags import FREQUENT_SAFETY_TAGS
from util.print_buffer import PrintBuffer

from classifiers.classifier import Classifier


class DnnClassifier(Classifier):
    model: tf.keras.Model
    classes: list[str]

    def __init__(self, model_paths: list[str]) -> None:
        super().__init__()
        self.model = keras.models.load_model(model_paths[0])
        with open(model_paths[1], "r") as f:
            self.classes = [x.strip() for x in f.readlines()]

    def get_classification(
        self, evaluation_dict: dict[str, int], print_buffer: PrintBuffer
    ) -> str:
        input_dict = {
            tag: tf.convert_to_tensor([reliability])
            for tag, reliability in evaluation_dict.items()
            if tag in FREQUENT_SAFETY_TAGS
        }
        predictions = self.model(input_dict)[0]
        return self.classes[np.argmax(predictions)]
