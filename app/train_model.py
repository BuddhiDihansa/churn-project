from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn_utils import ARTIFACT_DIR, DATA_PATH, METADATA_PATH, MODEL_PATH, build_metadata, build_model, load_dataframe, prepare_training_data


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataframe(DATA_PATH)
    X, y = prepare_training_data(dataframe)

    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "roc_auc" if y.nunique() > 1 else "accuracy"
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score="raise")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    model.fit(X_train, y_train)
    test_probability = model.predict_proba(X_test)[:, 1] if y.nunique() > 1 else None
    test_score = roc_auc_score(y_test, test_probability) if test_probability is not None else None

    model.fit(X, y)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    metadata = build_metadata(dataframe)
    metadata["scoring"] = scoring
    metadata["cv_mean"] = float(cv_scores.mean())
    metadata["cv_std"] = float(cv_scores.std())
    metadata["test_roc_auc"] = float(test_score) if test_score is not None else None

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Cross-validation {scoring}: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    if test_score is not None:
        print(f"Test ROC-AUC: {test_score:.4f}")


if __name__ == "__main__":
    main()
