import os
import pickle
from typing import Iterable, Union

import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from util.print_buffer import PrintBuffer

from classifiers.classifier import Classifier


class DecisionTreeClassifier(Classifier):
    model: tree.DecisionTreeClassifier
    feature_names:Iterable[str]

    def create_model(self, df: pd.DataFrame, tags: Iterable[str], **kwargs) -> None:
        classifier = tree.DecisionTreeClassifier(
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            ccp_alpha=kwargs.get("ccp_alpha", 0),
        )
        self.model = classifier.fit(
            df.drop(columns=[self.CLASS_COLUMN]).to_numpy(), df[self.CLASS_COLUMN]
        )
        self.feature_names=df.columns.drop(self.CLASS_COLUMN)

    def load_model(self, paths: Union[list[str], None]) -> None:
        if paths is None:
            raise TypeError("paths must not be None")

        with open(paths[0], "rb") as f:
            self.model = pickle.load(f)

    def export_model(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, "decision_tree.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        dot_data: str = tree.export_graphviz(
            self.model,
            feature_names=self.feature_names,
            class_names=self.model.classes_,
            filled=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        graph.render(
            outfile=os.path.join(output_dir, "decision_tree.png"), cleanup=True
        )

    def get_classification(
        self,
        evaluation_dict: dict[str, int],
        tags: Iterable[str],
        print_buffer: PrintBuffer,
    ) -> str:
        evaluations = [evaluation_dict[x] for x in tags]
        evaluations = np.reshape(evaluations, (1, -1))
        return self.model.predict(evaluations)[0]
