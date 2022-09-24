import argparse
import shutil
import os
from deepdanbooru import io as dd_io, extra as dd_extra, project as dd_project
from deepdanbooru.commands.evaluate import evaluate_image
from default_classifier import DefaultClassifier


def main():
    parser = argparse.ArgumentParser(
        description='Organize images into directories using deepdanbooru prediction.')
    parser.add_argument('project_dir')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    classifier = DefaultClassifier()
    process_images(args.project_dir, args.input_dir,
                   args.output_dir, classifier)


def process_images(project_dir, input_dir, output_dir, classifier):
    print(f'Importing images from {input_dir}...')
    file_pattern = '*.[Pp][Nn][Gg],*.[Jj][Pp][Gg],*.[Jj][Pp][Ee][Gg],*.[Gg][Ii][Ff]'
    input_image_paths = dd_io.get_image_file_paths_recursive(
        input_dir, file_pattern)
    input_image_paths = dd_extra.natural_sorted(input_image_paths)
    print(f'{len(input_image_paths)} files imported.')

    print('Loading model...')
    model = dd_project.load_model_from_project(
        project_dir, compile_model=False
    )

    print('Loading tags...')
    tags = dd_project.load_tags_from_project(project_dir)

    print('Evaluating...')
    for input_image_path in input_image_paths:
        evaluations = evaluate_image(input_image_path, model, tags, 0)
        evaluation_dict = dict(evaluations)
        image_name = os.path.basename(input_image_path)
        classification = classifier.get_classification(
            evaluation_dict, image_name)
        copy_image(input_image_path, output_dir, classification)


def copy_image(input_image_path, output_dir, classification):
    output_class_dir = os.path.join(output_dir, classification)
    os.makedirs(output_class_dir, exist_ok=True)
    shutil.copy2(input_image_path, output_class_dir)


if __name__ == '__main__':
    main()
