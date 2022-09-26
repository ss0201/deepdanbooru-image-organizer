from typing import Any

from deepdanbooru import extra as dd_extra
from deepdanbooru import io as dd_io
from deepdanbooru import project as dd_project
from deepdanbooru.commands.evaluate import (  # noqa: F401
    evaluate_image as evaluate_image,
)


def load_project(dir: str, verbose: bool) -> tuple[Any, list[str]]:
    if verbose:
        print("Loading model...")
    model = dd_project.load_model_from_project(dir, compile_model=False)

    if verbose:
        print("Loading tags...")
    tags = dd_project.load_tags_from_project(dir)

    return model, tags


def load_images(dir: str, verbose: bool) -> list[str]:
    if verbose:
        print(f"Importing images from {dir}...")
    file_pattern = "*.[Pp][Nn][Gg],*.[Jj][Pp][Gg],*.[Jj][Pp][Ee][Gg],*.[Gg][Ii][Ff]"
    paths = dd_io.get_image_file_paths_recursive(dir, file_pattern)
    paths = dd_extra.natural_sorted(paths)
    if verbose:
        print(f"{len(paths)} files imported.")

    return paths


def load_project_and_images(
    project_dir: str, image_dir: str, verbose: bool
) -> tuple[Any, list[str], list[str]]:
    model, tags = load_project(project_dir, verbose)
    image_paths = load_images(image_dir, verbose)
    return model, tags, image_paths
