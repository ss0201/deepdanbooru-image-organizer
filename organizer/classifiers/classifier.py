from abc import ABC, abstractmethod
from typing import Iterable, Union

import pandas as pd
from util.print_buffer import PrintBuffer


class Classifier(ABC):
    CLASS_COLUMN: str = "class"

    def create_model(self, df: pd.DataFrame, tags: Iterable[str], **kwargs) -> None:
        pass

    def export_model(self, output_dir: str) -> None:
        pass

    def load_model(self, paths: Union[list[str], None]) -> None:
        pass

    def set_parameters(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_classification(
        self, evaluation_dict: dict[str, float], print_buffer: PrintBuffer
    ) -> str:
        pass
