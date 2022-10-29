from util.print_buffer import PrintBuffer

from classifiers.classifier import Classifier


class SingleTagClassifier(Classifier):
    tag: str
    threshold: float
    classes: list[str]

    def set_parameters(self, **kwargs) -> None:
        tag = kwargs.get("tag", None)
        threshold = kwargs.get("threshold", None)
        classes = kwargs.get("classes", None)

        if not tag or not threshold or not classes:
            raise ValueError("tag, threshold, classes must be provided")
        if len(classes) != 2:
            raise ValueError("classes must have length 2")

        self.tag = tag
        self.threshold = threshold
        self.classes = classes

    def get_classification(
        self, evaluation_dict: dict[str, float], print_buffer: PrintBuffer
    ) -> str:
        return (
            self.classes[0]
            if evaluation_dict[self.tag] >= self.threshold
            else self.classes[1]
        )
