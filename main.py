import argparse
import shutil
import os
from deepdanbooru import io as dd_io, extra as dd_extra, project as dd_project
from deepdanbooru.commands.evaluate import evaluate_image


def main():
    parser = argparse.ArgumentParser(
        description='Organize images into directories using deepdanbooru prediction.')
    parser.add_argument('project_dir')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    classify_images(args.project_dir, args.input_dir, args.output_dir)


def classify_images(project_dir, input_dir, output_dir):
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
        classify_image(input_image_path, evaluation_dict, output_dir)


def classify_image(input_image_path, evaluation_dict, rating_output_dir):
    rating_keys = ['explicit', 'questionable', 'safe']
    ratings = [(key, evaluation_dict[f'rating:{key}']) for key in rating_keys]
    sorted_ratings = sorted(ratings, key=lambda x: x[1], reverse=True)
    best_rating = sorted_ratings[0]

    rating_output_dir = os.path.join(rating_output_dir, best_rating[0])
    os.makedirs(rating_output_dir, exist_ok=True)
    image_name = os.path.basename(input_image_path)
    output_file = os.path.join(rating_output_dir, image_name)
    shutil.copy2(input_image_path, output_file)

    print(f'{image_name}: {sorted_ratings}')


if __name__ == '__main__':
    main()
