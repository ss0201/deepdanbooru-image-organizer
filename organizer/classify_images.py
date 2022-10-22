import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any, Union

from classifiers import Classifier, classifier_factory
from data.evaluation import evaluate_image
from util import dd_adapter
from util.print_buffer import PrintBuffer


def main():
    parser = argparse.ArgumentParser(
        description="Organize images into directories using deepdanbooru prediction."
    )
    parser.add_argument("project_dir", help="DeepDanbooru project directory")
    parser.add_argument("input_dir", help="Input image directory")
    parser.add_argument("output_dir", help="Output image directory")
    parser.add_argument(
        "--model",
        required=False,
        help="Classification model name. Available: [forest, tree, dnn]",
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=False,
        help="Classification model paths. Required if --model is specified. \
            forest: [model_path], tree: [model_path], dnn: [model_path, tag_path]",
    )
    parser.add_argument(
        "--parallel", type=int, default=16, help="Number of parallel jobs. Default: 16"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without copying files to output dir"
    )
    args = parser.parse_args()

    process_images(
        args.project_dir,
        args.input_dir,
        args.output_dir,
        args.model,
        args.model_paths,
        args.parallel,
        args.dry_run,
    )


def process_images(
    project_dir: str,
    input_dir: str,
    output_dir: str,
    model_name: str,
    model_paths: Union[list[str], None],
    parallel: int,
    dry_run: bool,
) -> None:
    dd_model, dd_tags, input_image_paths = dd_adapter.load_project_and_images(
        project_dir, input_dir, True
    )
    classifier = classifier_factory.get_classifier(model_name)
    classifier.load_model(model_paths)

    print("Processing images...")
    with ThreadPoolExecutor(parallel) as executor:
        dd_semaphore = BoundedSemaphore()
        classifier_semaphore = BoundedSemaphore()
        futures = [
            executor.submit(
                process_image,
                input_image_path,
                dd_model,
                dd_tags,
                output_dir,
                dry_run,
                classifier,
                dd_semaphore,
                classifier_semaphore,
            )
            for input_image_path in input_image_paths
        ]
        for future in futures:
            _ = future.result()  # call result() so that exceptions are raised


def process_image(
    input_image_path: str,
    dd_model: Any,
    dd_tags: list[str],
    output_dir: str,
    dry_run: bool,
    classifier: Classifier,
    dd_semaphore: BoundedSemaphore,
    classifier_semaphore: BoundedSemaphore,
) -> None:
    print_buffer = PrintBuffer()
    with dd_semaphore:
        evaluation_dict = evaluate_image(input_image_path, dd_model, dd_tags)

    image_name = os.path.basename(input_image_path)
    print_buffer.add(f"* {image_name}")

    with classifier_semaphore:
        classification = classifier.get_classification(evaluation_dict, print_buffer)
    print_buffer.add(f"Class: {classification}\n")

    if not dry_run:
        copy_image(input_image_path, output_dir, classification)

    print_buffer.print()


def copy_image(input_image_path: str, output_dir: str, classification: str):
    output_class_dir = os.path.join(output_dir, classification)
    os.makedirs(output_class_dir, exist_ok=True)
    shutil.copy2(input_image_path, output_class_dir)


if __name__ == "__main__":
    main()
