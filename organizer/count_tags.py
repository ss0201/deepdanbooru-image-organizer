import argparse
import collections

from data import tag_list
from data.dd_cache import create_cache
from data.evaluation import evaluate_image
from util import dd_adapter


def main():
    parser = argparse.ArgumentParser(description="Count tags used in images.")
    parser.add_argument("project_dir", help="DeepDanbooru project directory")
    parser.add_argument("--input", nargs="+", help="Input image directories")
    parser.add_argument("--output", default=".", help="Output directory for tag list")
    parser.add_argument(
        "--cache",
        help="Tag prediction cache file. If the file does not exist, \
            a new file will be created. If not specified, do not use cache.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Tag confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Tag number limit. Default: 100"
    )
    args = parser.parse_args()

    common_tags = calc_common_tags(
        args.project_dir, args.input, args.cache, args.threshold, args.limit
    )
    print(common_tags)
    tag_list.write(args.output, [x[0] for x in common_tags])


def calc_common_tags(
    project_dir: str,
    image_dirs: list[str],
    cache_dir: str,
    threshold: float,
    limit: int,
) -> list[tuple[str, int]]:
    model, tags = dd_adapter.load_project(project_dir, True)
    cache = create_cache(cache_dir)

    image_paths = []
    for image_dir in image_dirs:
        image_paths.extend(dd_adapter.load_images(image_dir, True))

    print("Counting tags...")
    tag_counter: collections.Counter[str] = collections.Counter()
    for image_path in image_paths:
        evaluation_dict = evaluate_image(image_path, model, tags, threshold, cache)
        tag_counter.update(evaluation_dict.keys())

    return tag_counter.most_common(limit)


if __name__ == "__main__":
    main()
