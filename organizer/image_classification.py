import argparse
import os
import shutil
from pathlib import Path
from typing import Union

from classifiers import Classifier, DefaultClassifier
from classifiers.decision_tree_classifier import DecisionTreeClassifier
from util import dd_adapter


def main():
    parser = argparse.ArgumentParser(
        description="Organize images into directories using deepdanbooru prediction."
    )
    parser.add_argument("project_dir")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--model", required=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    classifier = get_classifier(args.model)
    process_images(
        args.project_dir, args.input_dir, args.output_dir, args.dry_run, classifier
    )


def get_classifier(model: Union[str, None]) -> Classifier:
    if model is None:
        return DefaultClassifier()

    model_name = Path(model).stem
    if model_name == "decision_tree":
        return DecisionTreeClassifier(model)

    raise Exception(f"Invalid model: {model}")


def process_images(
    project_dir: str,
    input_dir: str,
    output_dir: str,
    dry_run: bool,
    classifier: Classifier,
) -> None:
    model, tags, input_image_paths = dd_adapter.load_project_and_images(
        project_dir, input_dir, True
    )

    print("Evaluating...")
    for input_image_path in input_image_paths:
        evaluations = dd_adapter.evaluate_image(input_image_path, model, tags, 0)
        evaluation_dict = dict(evaluations)
        image_name = os.path.basename(input_image_path)
        print(f"* {image_name}")
        classification = classifier.get_classification(evaluation_dict)
        print(f"Class: {classification}")
        if not dry_run:
            copy_image(input_image_path, output_dir, classification)
        print()


def copy_image(input_image_path: str, output_dir: str, classification: str):
    output_class_dir = os.path.join(output_dir, classification)
    os.makedirs(output_class_dir, exist_ok=True)
    shutil.copy2(input_image_path, output_class_dir)


if __name__ == "__main__":
    main()
