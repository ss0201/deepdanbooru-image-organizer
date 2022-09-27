import argparse
import os
import pickle

import graphviz
import pandas as pd
from sklearn import tree

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
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-leaf", type=int, default=1)
    parser.add_argument("--ccp-alpha", type=float, default=0)

    args = parser.parse_args()

    tags = sorted(set(EXPLICIT_TAGS + QUESTIONABLE_TAGS + SAFE_TAGS))

    if args.dataframe:
        df: pd.DataFrame = pd.read_pickle(args.dataframe)
    else:
        print("Creating dataframe...")
        df = dataframe.create(args.project_dir, args.input_dirs, tags, CLASS_COLUMN)
        print(df)
        dataframe.export(df, args.output_dir)

    print("Creating decision tree...")
    classifier = create_decision_tree(df, args.max_depth, args.min_leaf, args.ccp_alpha)
    export_decision_tree(classifier, df, args.output_dir)

    print("Done.")


def create_decision_tree(
    df: pd.DataFrame, max_depth: int, min_samples_leaf: int, ccp_alpha: float
) -> tree.DecisionTreeClassifier:
    classifier = tree.DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha
    )
    return classifier.fit(df.drop(columns=[CLASS_COLUMN]).to_numpy(), df[CLASS_COLUMN])


def export_decision_tree(
    classifier: tree.DecisionTreeClassifier, df: pd.DataFrame, output_dir: str
):
    with open(os.path.join(output_dir, "decision_tree.pkl"), "wb") as f:
        pickle.dump(classifier, f)

    dot_data = tree.export_graphviz(
        classifier,
        feature_names=df.columns.drop(CLASS_COLUMN),
        class_names=classifier.classes_,
        filled=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(outfile=os.path.join(output_dir, "decision_tree.png"), cleanup=True)


if __name__ == "__main__":
    main()
