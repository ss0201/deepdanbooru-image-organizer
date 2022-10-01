from collections import namedtuple

from util.print_buffer import PrintBuffer

from classifiers.classifier import Classifier


class DefaultClassifier(Classifier):
    def get_classification(
        self, evaluation_dict: dict[str, int], print_buffer: PrintBuffer
    ) -> str:
        cls = Class()

        explicit_keys = [Keyword.NUDE]
        if self.contains_filtered_keys(explicit_keys, evaluation_dict, print_buffer):
            cls.try_set(Keyword.EXPLICIT)

        questionable_keys = [
            Keyword.SWIMSUIT,
            Keyword.BIKINI,
            Keyword.UNDERWEAR,
            Keyword.LEOTARD,
            Keyword.ASS,
        ]
        if self.contains_filtered_keys(
            questionable_keys, evaluation_dict, print_buffer
        ):
            cls.try_set(Keyword.QUESTIONABLE)

        rating_keys = [Keyword.EXPLICIT, Keyword.QUESTIONABLE, Keyword.SAFE]
        RatingEvaluation = namedtuple("RatingEvaluation", ["rating_key", "reliability"])
        ratings = [
            RatingEvaluation(key, evaluation_dict[f"rating:{key}"])
            for key in rating_keys
        ]
        sorted_ratings = sorted(ratings, key=lambda x: x.reliability, reverse=True)
        print_buffer.add(sorted_ratings)

        best_rating = sorted_ratings[0]
        if best_rating.reliability < 0.5:
            cls.try_set(Keyword.UNKNOWN)
        else:
            cls.try_set(best_rating.rating_key)

        return cls.value

    def contains_filtered_keys(
        self,
        keys: list[str],
        evaluation_dict: dict[str, int],
        print_buffer: PrintBuffer,
    ) -> bool:
        for key in keys:
            reliability = evaluation_dict[key]
            if reliability > 0.5:
                print_buffer.add(f"'{key}' {reliability}")
                return True
        return False


class Keyword:
    EXPLICIT = "explicit"
    QUESTIONABLE = "questionable"
    SAFE = "safe"
    UNKNOWN = "unknown"
    NUDE = "nude"
    SWIMSUIT = "swimsuit"
    BIKINI = "bikini"
    UNDERWEAR = "underwear"
    LEOTARD = "leotard"
    ASS = "ass"


class Class:
    value = Keyword.UNKNOWN
    CLASS_PRIORITY = {
        Keyword.EXPLICIT: 1,
        Keyword.QUESTIONABLE: 2,
        Keyword.SAFE: 3,
        Keyword.UNKNOWN: 4,
    }

    def try_set(self, value: str):
        if Class.CLASS_PRIORITY[value] < Class.CLASS_PRIORITY[self.value]:
            self.value = value
