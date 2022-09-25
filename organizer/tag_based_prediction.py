import argparse
import graphviz
import os
import numpy as np
import pandas as pd
from util import dd_adapter
from sklearn import tree
from util.frequent_safety_tags import *

CLASS_COLUMN = 'class'


def main():
    parser = argparse.ArgumentParser(
        description='Predict classification based on tags.')
    parser.add_argument('project_dir')
    parser.add_argument('input_dirs', nargs='+')
    parser.add_argument('--output', default='.', dest='output_dir')
    parser.add_argument('--dataframe', required=False)
    parser.add_argument('--max-depth', type=int, required=False)
    parser.add_argument('--min-leaf', type=int, required=False)

    args = parser.parse_args()

    if args.dataframe:
        df: pd.DataFrame = pd.read_pickle(args.dataframe)
    else:
        print('Creating dataframe...')
        tags = list(set(EXPLICIT_TAGS + QUESTIONABLE_TAGS + SAFE_TAGS))
        df = create_dataframe(args.project_dir, args.input_dirs, tags)
        print(df)
        output_dataframe(df, args.output_dir)

    print('Creating decision tree...')
    classifier = create_decision_tree(
        df, args.max_depth, args.min_leaf)
    output_decision_tree(classifier, df, args.output_dir)

    print('Done.')


def create_dataframe(project_dir: str, image_dirs: list[str], tags: list[str]) -> pd.DataFrame:
    model, _ = dd_adapter.load_project(project_dir, True)

    print('Predicting tags...')
    rows = np.empty((0, 2 + len(tags)), dtype=str)
    for image_dir in image_dirs:
        classification = os.path.basename(image_dir)
        image_paths = dd_adapter.load_images(image_dir, True)
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            evaluations = dd_adapter.evaluate_image(
                image_path, model, tags, 0)
            row = [image_name, classification] + [x[1] for x in evaluations]
            rows = np.append(rows, [row], axis=0)
    return pd.DataFrame(rows[:, 1:], index=rows[:, 0],
                        columns=[CLASS_COLUMN] + tags)


def output_dataframe(df: pd.DataFrame, output_dir: str) -> None:
    df.to_pickle(os.path.join(output_dir, 'dataframe.pkl'))


def create_decision_tree(df: pd.DataFrame, max_depth: int, min_samples_leaf: int) -> tree.DecisionTreeClassifier:
    classifier = tree.DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    return classifier.fit(df.drop([CLASS_COLUMN], axis=1).to_numpy(), df[CLASS_COLUMN])


def output_decision_tree(classifier: tree.DecisionTreeClassifier, df: pd.DataFrame, output_dir: str):
    dot_data = tree.export_graphviz(
        classifier, feature_names=df.columns.drop(CLASS_COLUMN), class_names=df[CLASS_COLUMN].unique(), filled=True)
    graph = graphviz.Source(dot_data)
    graph.render(outfile=os.path.join(
        output_dir, 'decision_tree.png'), cleanup=True)


if __name__ == '__main__':
    main()
