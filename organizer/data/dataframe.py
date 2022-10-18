import os
from typing import Any, Iterable, Union

import numpy as np
import pandas as pd
from util import dd_adapter

import data.dd_cache as dd_cache


def create(
    project_dir: str,
    image_dirs: Iterable[str],
    cache_path: Union[str, None],
    limited_tags: list[str],
    class_column: str,
) -> pd.DataFrame:
    model, project_tags = dd_adapter.load_project(project_dir, True)

    cache = None
    if cache_path:
        print("Loading tag prediction cache...")
        cache = dd_cache.load_or_create_cache(cache_path)

    print("Predicting tags...")
    rows = np.empty((0, 2 + len(limited_tags)))
    for image_dir in image_dirs:
        classification = os.path.basename(os.path.normpath(image_dir))
        image_paths = dd_adapter.load_images(image_dir, True)
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            evaluation_dict = evaluate(model, project_tags, image_path, cache)
            tag_reliabilities = [evaluation_dict[x] for x in limited_tags]
            row = [image_name, classification] + tag_reliabilities
            rows = np.append(rows, [row], axis=0)

    if cache is not None and cache_path:
        print("Saving tag prediction cache...")
        dd_cache.save_cache(cache, cache_path)

    print("Converting to dataframe...")
    df = pd.DataFrame(
        rows[:, 1:], index=rows[:, 0], columns=[class_column] + limited_tags
    )
    df: pd.DataFrame = df.apply(pd.to_numeric, errors="ignore")  # type: ignore
    df = df.astype({"class": pd.StringDtype()})
    return df


def evaluate(
    model: Any,
    project_tags: list[str],
    image_path: str,
    cache: Union[dd_cache.CacheType, None],
) -> dd_cache.CacheValueType:
    if cache is not None:
        evaluation_dict = dd_cache.get_cached_evaluations(image_path, cache)
        if evaluation_dict:
            return evaluation_dict

    evaluations = dd_adapter.evaluate_image(image_path, model, project_tags, 0)
    evaluation_dict = dict(evaluations)
    if cache is not None:
        dd_cache.cache_evaluations(image_path, evaluation_dict, cache)
    return evaluation_dict


def export(df: pd.DataFrame, output_dir: str) -> None:
    df.to_pickle(os.path.join(output_dir, "dataframe.pkl"))
