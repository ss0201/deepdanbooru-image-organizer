from threading import Semaphore
from typing import Any, Union

from util import dd_adapter

from data.dd_cache import DDCache


def evaluate_image(
    image_path: str,
    model: Any,
    project_tags: list[str],
    threshold: float = 0,
    cache: Union[DDCache, None] = None,
    semaphore: Union[Semaphore, None] = None,
) -> dict[str, float]:
    if cache is not None:
        evaluations = cache.get_cached_evaluations(image_path)
        if evaluations:
            return {
                tag: evaluation
                for tag, evaluation in zip(project_tags, evaluations)
                if evaluation >= threshold
            }

    if semaphore is not None:
        semaphore.acquire()
    try:
        evaluations = dd_adapter.evaluate_image(image_path, model, project_tags, 0)
        evaluation_dict = dict(evaluations)
    finally:
        if semaphore is not None:
            semaphore.release()

    if cache is not None:
        cache.cache_evaluations(
            image_path, [float(evaluation_dict[x]) for x in project_tags]
        )
    return {
        tag: evaluation
        for tag, evaluation in evaluation_dict.items()
        if evaluation >= threshold
    }
