import argparse

import pandas as pd
from sklearn import ensemble


def main():
    parser = argparse.ArgumentParser(
        description="Output the most important features of a random forest model."
    )
    parser.add_argument("model")
    args = parser.parse_args()

    output_features_importance(args.model)


def output_features_importance(model_path: str):
    model: ensemble.RandomForestClassifier = pd.read_pickle(model_path)

    importances = [
        (importance, name)
        for importance, name in zip(
            model.feature_importances_, model.feature_names_in_  # type: ignore
        )
    ]
    for importance, name in sorted(importances):
        print(f"{name:30} {importance:10.5f}")


if __name__ == "__main__":
    main()
