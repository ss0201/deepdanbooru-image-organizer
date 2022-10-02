from typing import Iterable


def read(paths: list[str]) -> list[str]:
    tags = set()
    for path in paths:
        with open(path, "r") as file:
            tags.update(file.read().splitlines())
    return sorted(tags)


def write(path: str, tags: Iterable[str]) -> None:
    with open(path, "w") as file:
        file.writelines(tags)
