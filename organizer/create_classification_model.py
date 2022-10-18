import argparse
import os
from typing import Iterable, Union

import pandas as pd

from classifiers import classifier_factory
from data import dataframe, tag_list

CLASS_COLUMN = "class"


def main():
    parser = argparse.ArgumentParser(
        description="Create a model to predict classification by danbooru tags."
    )
    parser.add_argument("project_dir")
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--output", default=".")
    parser.add_argument("--model")
    parser.add_argument("--dataframe")
    parser.add_argument("--cache")
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Only for tree based models."
    )
    parser.add_argument(
        "--min-leaf", type=int, default=1, help="Only for tree based models."
    )
    parser.add_argument(
        "--ccp-alpha", type=float, default=0, help="Only for tree based models."
    )
    args = parser.parse_args()

    create_model(
        args.project_dir,
        args.tags,
        args.input,
        args.output,
        args.model,
        args.dataframe,
        args.cache,
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
        ccp_alpha=args.ccp_alpha,
    )


def create_model(
    project_dir: str,
    tag_paths: Iterable[str],
    input_dirs: Iterable[str],
    output_dir: str,
    model_name: Union[str, None],
    dataframe_path: Union[str, None],
    cache_path: Union[str, None],
    **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    tags = tag_list.read(tag_paths)

    if dataframe_path:
        print("Loading dataframe...")
        df: pd.DataFrame = pd.read_pickle(dataframe_path)
    else:
        print("Creating dataframe...")
        df = dataframe.create(project_dir, input_dirs, cache_path, tags, CLASS_COLUMN)
        print(df)
        dataframe.export(df, output_dir)

    print("Creating classification model...")
    classifier = classifier_factory.get_classifier(model_name)
    classifier.create_model(df, tags, **kwargs)
    classifier.export_model(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
