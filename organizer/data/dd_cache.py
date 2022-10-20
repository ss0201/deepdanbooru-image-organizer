import bz2
import hashlib
import os
import pickle
from typing import Union

EvaluationDictType = dict[str, float]
MemoryType = dict[str, EvaluationDictType]


class EvaluationCache:
    memory: MemoryType
    is_dirty: bool

    def __init__(self, memory: MemoryType = {}):
        self.memory = memory
        self.is_dirty = False

    def get(self, key: str) -> Union[EvaluationDictType, None]:
        return self.memory.get(key)

    def set(self, key: str, value: EvaluationDictType) -> None:
        self.memory[key] = value
        self.is_dirty = True


class DDCache:
    cache_collection: dict[str, EvaluationCache]
    cache_dir: str

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.cache_collection = {}
        os.makedirs(cache_dir, exist_ok=True)

    def get_cached_evaluations(
        self, image_path: str
    ) -> Union[EvaluationDictType, None]:
        cache_key, image_key = self.__get_cache_key_and_image_key(image_path)
        cache = self.__get_cache(cache_key)
        return cache.get(image_key)

    def cache_evaluations(
        self, image_path: str, evaluations: EvaluationDictType,
    ) -> None:
        cache_key, image_key = self.__get_cache_key_and_image_key(image_path)
        cache = self.__get_cache(cache_key)
        cache.set(image_key, evaluations)

    def save(self) -> None:
        for cache_key, cache in self.cache_collection.items():
            if cache.is_dirty:
                self.__save_cache_file(cache, os.path.join(self.cache_dir, cache_key))

    def __get_cache(self, cache_key) -> EvaluationCache:
        cache = self.cache_collection.get(cache_key)
        if cache is None:
            cache = self.__load_cache_file(os.path.join(self.cache_dir, cache_key))
            self.cache_collection[cache_key] = cache
        return cache

    def __get_cache_key_and_image_key(self, image_path: str) -> tuple[str, str]:
        image_key = self.__get_md5(image_path)
        return image_key[:2], image_key

    def __get_md5(self, image_path: str) -> str:
        md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def __load_cache_file(self, cache_path: str) -> EvaluationCache:
        if os.path.isfile(cache_path):
            with bz2.BZ2File(cache_path, "rb") as f:
                return EvaluationCache(pickle.load(f))
        else:
            return EvaluationCache()

    def __save_cache_file(self, cache: EvaluationCache, cache_path: str) -> None:
        with bz2.BZ2File(cache_path, "wb") as f:
            pickle.dump(cache, f)
