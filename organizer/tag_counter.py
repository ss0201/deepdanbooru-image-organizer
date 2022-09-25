import argparse
import collections
import pandas as pd
from util import dd_adapter
from util.frequent_safety_tags import *


def main():
    parser = argparse.ArgumentParser(
        description='Count tags used in images.')
    parser.add_argument('project_dir')
    parser.add_argument('input_dir')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    count_tags(args.project_dir, args.input_dir, args.threshold, args.limit)


def count_tags(project_dir: str, image_dir: list[str], threshold: float, limit: int) -> pd.DataFrame:
    model, tags = dd_adapter.load_project(project_dir, True)
    image_paths = dd_adapter.load_images(image_dir, True)

    print('Counting tags...')
    tag_counter = collections.Counter()
    for image_path in image_paths:
        evaluations = dd_adapter.evaluate_image(
            image_path, model, tags, threshold)
        tag_counter.update([x[0] for x in evaluations])

    most_common = tag_counter.most_common(limit)
    print(most_common)
    print([x[0] for x in most_common])


if __name__ == '__main__':
    main()
