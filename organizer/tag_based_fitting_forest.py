import argparse
import os
import pickle

import pandas as pd
from sklearn import ensemble

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
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-leaf", type=int, default=1)
    parser.add_argument("--ccp-alpha", type=float, default=0)
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

    print("Creating model...")
    classifier = create_model(df, args.max_depth, args.min_leaf, args.ccp_alpha)
    export_model(classifier, args.output_dir)

    print("Done.")


def create_model(
    df: pd.DataFrame, max_depth: int, min_samples_leaf: int, ccp_alpha: float
) -> ensemble.RandomForestClassifier:
    classifier = ensemble.RandomForestClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha
    )
    return classifier.fit(df.drop(columns=[CLASS_COLUMN]).to_numpy(), df[CLASS_COLUMN])


def export_model(classifier: ensemble.RandomForestClassifier, output_dir: str):
    with open(os.path.join(output_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(classifier, f)


if __name__ == "__main__":
    main()
