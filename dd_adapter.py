from typing import Any
from deepdanbooru import io as dd_io, extra as dd_extra, project as dd_project
from deepdanbooru.commands.evaluate import evaluate_image as evaluate_image


def load_images_and_project(project_dir: str, input_dir: str, verbose: bool) -> tuple[Any, list[str], list[str]]:
    if verbose:
        print('Loading model...')
    model = dd_project.load_model_from_project(
        project_dir, compile_model=False
    )

    if verbose:
        print('Loading tags...')
    tags = dd_project.load_tags_from_project(project_dir)

    if verbose:
        print(f'Importing images from {input_dir}...')
    file_pattern = '*.[Pp][Nn][Gg],*.[Jj][Pp][Gg],*.[Jj][Pp][Ee][Gg],*.[Gg][Ii][Ff]'
    input_image_paths = dd_io.get_image_file_paths_recursive(
        input_dir, file_pattern)
    input_image_paths = dd_extra.natural_sorted(input_image_paths)
    if verbose:
        print(f'{len(input_image_paths)} files imported.')

    return model, tags, input_image_paths
