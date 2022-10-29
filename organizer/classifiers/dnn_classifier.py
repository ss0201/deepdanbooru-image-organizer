# type: ignore

import os
from typing import Iterable, Union

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import Normalization
from util.print_buffer import PrintBuffer

from classifiers.classifier import Classifier


class DnnClassifier(Classifier):
    model: tf.keras.Model
    classes: list[str]

    def create_model(self, df: pd.DataFrame, tags: Iterable[str], **kwargs) -> None:
        self.classes = df[self.CLASS_COLUMN].unique().tolist()

        val_df = df.sample(frac=0.2)
        train_df = df.drop(val_df.index)
        train_ds = self.dataframe_to_dataset(train_df, self.classes)
        val_ds = self.dataframe_to_dataset(val_df, self.classes)
        train_ds = train_ds.batch(32)
        val_ds = val_ds.batch(32)

        tags_input = [keras.Input(shape=(1,), name=tag, dtype="float") for tag in tags]
        all_inputs = tags_input

        tags_encoded = [
            self.encode_numerical_feature(tag_input, tag_input.name, train_ds)
            for tag_input in tags_input
        ]
        all_features = layers.concatenate(tags_encoded)

        x = layers.Dense(32, activation="relu")(all_features)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(len(self.classes), activation="softmax")(x)
        model = keras.Model(all_inputs, output)
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, epochs=50, validation_data=val_ds)
        self.model = model

    def export_model(self, output_dir: str):
        self.model.save(os.path.join(output_dir, "dnn.h5"))
        with open(os.path.join(output_dir, "dnn_class.txt"), "w") as f:
            f.write("\n".join(self.classes))

    def load_model(self, paths: Union[list[str], None]) -> None:
        if paths is None:
            raise TypeError("paths must not be None")

        self.model = keras.models.load_model(paths[0])
        with open(paths[1], "r") as f:
            self.classes = f.read().splitlines()

    def get_classification(
        self, evaluation_dict: dict[str, float], print_buffer: PrintBuffer
    ) -> str:
        tags = [x.name for x in self.model.inputs]
        input_dict = {
            tag: tf.convert_to_tensor([reliability])
            for tag, reliability in evaluation_dict.items()
            if tag in tags
        }
        predictions = self.model(input_dict)[0]
        return self.classes[np.argmax(predictions)]

    def dataframe_to_dataset(self, df: pd.DataFrame, classes: list[str]):
        df = df.copy()
        labels = df.pop("class")
        labels = [classes.index(x) for x in labels]
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        ds = ds.shuffle(buffer_size=len(df))
        return ds

    def encode_numerical_feature(self, feature, name, dataset):
        normalizer = Normalization()

        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        normalizer.adapt(feature_ds)

        encoded_feature = normalizer(feature)
        return encoded_feature
