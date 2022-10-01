from abc import ABC, abstractmethod

from util.print_buffer import PrintBuffer


class Classifier(ABC):
    @abstractmethod
    def get_classification(
        self, evaluation_dict: dict[str, int], print_buffer: PrintBuffer
    ) -> str:
        pass
