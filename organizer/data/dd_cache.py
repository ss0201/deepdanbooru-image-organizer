import bz2
import hashlib
import os
import pickle
from typing import Union

CacheValueType = dict[str, float]
CacheType = dict[str, CacheValueType]


def load_or_create_cache(cache_path: str) -> CacheType:
    if os.path.isfile(cache_path):
        with bz2.BZ2File(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        return {}


def save_cache(cache: CacheType, cache_path: str) -> None:
    with bz2.BZ2File(cache_path, "wb") as f:
        pickle.dump(cache, f)


def cache_evaluations(
    image_path: str, evaluations: CacheValueType, cache: CacheType,
) -> None:
    key = get_md5(image_path)
    cache[key] = evaluations


def get_cached_evaluations(
    image_path: str, cache: CacheType
) -> Union[CacheValueType, None]:
    key = get_md5(image_path)
    return cache.get(key)


def get_md5(image_path: str) -> str:
    md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()
