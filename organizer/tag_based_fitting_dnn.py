import argparse
import os
from typing import Iterable

import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import IntegerLookup, Normalization, StringLookup
from tensorflow import keras

from data import dataframe, tag_list

CLASS_COLUMN = "class"


def main():
    parser = argparse.ArgumentParser(
        description="Predict classification based on tags."
    )
    parser.add_argument("project_dir")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--input", nargs="+", required=False, dest="input_dirs")
    parser.add_argument("--output", default=".", dest="output_dir")
    parser.add_argument("--dataframe", required=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tags = tag_list.read(args.tags)

    if args.dataframe:
        df: pd.DataFrame = pd.read_pickle(args.dataframe)
    else:
        print("Creating dataframe...")
        df = dataframe.create(args.project_dir, args.input_dirs, tags, CLASS_COLUMN)
        print(df)
        dataframe.export(df, args.output_dir)

    classes = df[CLASS_COLUMN].unique().tolist()
    model = create_dnn_model(df, tags, classes)
    export_model(model, classes, args.output_dir)

    print("Done.")


def create_dnn_model(
    df: pd.DataFrame, tags: Iterable[str], classes: list[str]
) -> tf.keras.Model:
    val_df = df.sample(frac=0.2)
    train_df = df.drop(val_df.index)
    train_ds = dataframe_to_dataset(train_df, classes)
    val_ds = dataframe_to_dataset(val_df, classes)
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
    output = layers.Dense(len(classes), activation="softmax")(x)
    model = keras.Model(all_inputs, output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=50, validation_data=val_ds)
    return model


def dataframe_to_dataset(df: pd.DataFrame, classes: list[str]):
    df = df.copy()
    labels = df.pop("class")
    labels = [classes.index(x) for x in labels]
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))
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


def export_model(model: tf.keras.Model, classes: Iterable[str], output_dir: str):
    model.save(os.path.join(output_dir, "dnn.h5"))
    with open(os.path.join(output_dir, "dnn_class.txt"), "w") as f:
        f.writelines(classes)


if __name__ == "__main__":
    main()
