import argparse
import os
from typing import Iterable

import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import IntegerLookup, Normalization, StringLookup
from tensorflow import keras

from data import dataframe
from util.frequent_safety_tags import EXPLICIT_TAGS, QUESTIONABLE_TAGS, SAFE_TAGS

CLASS_COLUMN = "class"


def main():
    parser = argparse.ArgumentParser(
        description="Predict classification based on tags."
    )
    parser.add_argument("project_dir")
    parser.add_argument("--input_dirs", nargs="+", required=False)
    parser.add_argument("--output", default=".", dest="output_dir")
    parser.add_argument("--dataframe", required=False)

    args = parser.parse_args()

    tags = sorted(set(EXPLICIT_TAGS + QUESTIONABLE_TAGS + SAFE_TAGS))

    if args.dataframe:
        df: pd.DataFrame = pd.read_pickle(args.dataframe)
    else:
        print("Creating dataframe...")
        df = dataframe.create(args.project_dir, args.input_dirs, tags, CLASS_COLUMN)
        print(df)
        dataframe.export(df, args.output_dir)

    model = create_dnn_model(df, tags)
    export_model(model, args.output_dir)

    print("Done.")


def create_dnn_model(df: pd.DataFrame, tags: Iterable[str]) -> tf.keras.Model:
    num_classes = df[CLASS_COLUMN].unique().size

    val_df = df.sample(frac=0.2)
    train_df = df.drop(val_df.index)
    train_ds = dataframe_to_dataset(train_df, num_classes)
    val_ds = dataframe_to_dataset(val_df, num_classes)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    tags_input = [keras.Input(shape=(1,), name=tag, dtype="float") for tag in tags]
    all_inputs = tags_input

    tags_encoded = [
        encode_numerical_feature(tag_input, tag_input.name, train_ds)
        for tag_input in tags_input
    ]
    all_features = layers.concatenate(tags_encoded)

    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(all_inputs, output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=50, validation_data=val_ds)
    return model


def dataframe_to_dataset(df: pd.DataFrame, num_classes):
    df = df.copy()
    labels = df.pop("class")
    labels = [{"explicit": 2, "questionable": 1, "safe": 0}[x] for x in labels]
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    return ds


def encode_numerical_feature(feature, name, dataset):
    normalizer = Normalization()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    lookup = lookup_class(output_mode="binary")

    print(dataset)
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    lookup.adapt(feature_ds)

    encoded_feature = lookup(feature)
    return encoded_feature


def export_model(model: tf.keras.Model, output_dir: str):
    model.save(os.path.join(output_dir, "dnn.h5"))


if __name__ == "__main__":
    main()
