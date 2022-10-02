from abc import ABC, abstractmethod
from typing import Iterable

from util.print_buffer import PrintBuffer


class Classifier(ABC):
    @abstractmethod
    def get_classification(
        self,
        evaluation_dict: dict[str, int],
        tags: Iterable[str],
        print_buffer: PrintBuffer,
    ) -> str:
        pass
