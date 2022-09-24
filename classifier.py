from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def get_classification(self, evaluation_dict: dict[str, int]) -> str:
        pass
