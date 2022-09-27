import os
from typing import Iterable

import numpy as np
import pandas as pd
from util import dd_adapter


def create(
    project_dir: str,
    image_dirs: list[str],
    limited_tags: Iterable[str],
    class_column: str,
) -> pd.DataFrame:
    model, project_tags = dd_adapter.load_project(project_dir, True)

    print("Predicting tags...")
    rows = np.empty((0, 2 + len(limited_tags)))
    for image_dir in image_dirs:
        classification = os.path.basename(image_dir)
        image_paths = dd_adapter.load_images(image_dir, True)
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            evaluations = dd_adapter.evaluate_image(image_path, model, project_tags, 0)
            limited_evaluations = sorted(
                filter(lambda x: x[0] in limited_tags, evaluations)
            )
            tag_reliabilities = [x[1] for x in limited_evaluations]
            row = [image_name, classification] + tag_reliabilities
            rows = np.append(rows, [row], axis=0)

    df = pd.DataFrame(
        rows[:, 1:], index=rows[:, 0], columns=[class_column] + limited_tags
    )
    df: pd.DataFrame = df.apply(pd.to_numeric, errors="ignore")
    df = df.astype({"class": pd.StringDtype()})
    return df


def export(df: pd.DataFrame, output_dir: str) -> None:
    df.to_pickle(os.path.join(output_dir, "dataframe.pkl"))
