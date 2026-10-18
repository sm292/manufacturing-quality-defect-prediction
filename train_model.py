
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
MAX_SELECTED_FEATURES = 40

# Candidate probability thresholds for defect detection.
THRESHOLDS = np.arange(0.05, 0.51, 0.01)


# --------------------------------------------------
# 2. LOAD AND PREPARE THE DATASET
# --------------------------------------------------

def load_data():
    feature_path = RAW_DIR / "secom.data"
    label_path = RAW_DIR / "secom_labels.data"

    missing_files = [
        str(path)
        for path in [feature_path, label_path]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing dataset files: "
            + ", ".join(missing_files)
            + "\nRun: python download_data.py"
        )

    print("Loading dataset...")

    X = pd.read_csv(
        feature_path,
        sep=r"\s+",
        header=None,
        na_values="NaN",
        engine="python",
    )

    labels = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if len(X) != len(labels):
        raise ValueError(
            f"Feature rows ({len(X)}) do not match "
            f"label rows ({len(labels)})."
        )

    # UCI SECOM labels:
    # -1 = pass, 1 = fail.
    raw_labels = pd.to_numeric(labels.iloc[:, 0])

    if not raw_labels.isin([-1, 1]).all():
        raise ValueError(
            "Unexpected label values. Expected -1 and 1."
        )

    y = (raw_labels == 1).astype(int)

    # Parse the timestamp fields where available.
    if labels.shape[1] >= 3:
        timestamp_text = (
            labels.iloc[:, 1].astype(str)
            + " "
            + labels.iloc[:, 2].astype(str)
        )
    elif labels.shape[1] >= 2:
        timestamp_text = labels.iloc[:, 1].astype(str)
    else:
        timestamp_text = pd.Series(
            pd.NaT, index=labels.index
        )

    timestamps = pd.to_datetime(
        timestamp_text,
        errors="coerce",
    )

    # Sort chronologically only when every timestamp is valid.
    if timestamps.notna().all():
        order = np.argsort(timestamps.to_numpy())

        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        timestamps = timestamps.iloc[order].reset_index(
            drop=True
        )
        print("Rows sorted by timestamp.")
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        timestamps = timestamps.reset_index(drop=True)
        print(
            "Warning: complete timestamps unavailable; "
            "using the dataset's original row order."
        )

    # Stable sensor names for saved feature importance.
    X.columns = [
        f"sensor_{i:03d}"
        for i in range(X.shape[1])
    ]

    print(f"Rows: {len(X)}")
    print(f"Sensor features: {X.shape[1]}")
    print(f"Passes: {(y == 0).sum()}")
    print(f"Failures: {(y == 1).sum()}")
    print(f"Failure rate: {y.mean():.2%}")

    return X, y, timestamps


# --------------------------------------------------
# 3. BUILD THE MODEL PIPELINE
# --------------------------------------------------

def create_model(n_features):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "remove_constant",
            VarianceThreshold(),
        ),
        (
            "select_features",
            SelectKBest(
                score_func=f_classif,
                k=n_features,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# --------------------------------------------------
# 4. SELECT A DECISION THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

def select_threshold(y_true, probabilities):
    """
    Choose the threshold with the highest validation F1-score.

    The independent test set is not used to select this
    threshold. In a production project, threshold selection
    should also account for the relative cost of false
    negatives and false positives.
    """

    best_threshold = 0.50
    best_f1 = -1.0

    threshold_results = []

    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        threshold_results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        })

        # Deterministic tie-break: retain the higher
        # threshold when F1 scores are equal.
        if (
            f1 > best_f1
            or (
                np.isclose(f1, best_f1)
                and threshold > best_threshold
            )
        ):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1, threshold_results


# --------------------------------------------------
# 5. TRAIN, VALIDATE, AND EVALUATE
# --------------------------------------------------

def main():
    X, y, timestamps = load_data()

    if y.nunique() != 2:
        raise ValueError(
            "Training requires both pass and fail examples."
        )

    # Reserve the latest 20% for final evaluation.
    split_index = int(len(X) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "The chronological split does not contain both "
            "classes in training and test data. Inspect the "
            "timestamp ordering and class distribution."
        )

    # Separate validation data from the training period.
    # Stratification helps preserve the rare failure class.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    n_features = min(
        MAX_SELECTED_FEATURES,
        X.shape[1],
    )

    print("\nTraining validation model...")

    validation_model = create_model(n_features)
    validation_model.fit(X_fit, y_fit)

    validation_probabilities = (
        validation_model.predict_proba(X_val)[:, 1]
    )

    best_threshold, validation_f1, threshold_results = (
        select_threshold(
            y_val,
            validation_probabilities,
        )
    )

    print(f"Validation rows: {len(X_val)}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation F1-score: {validation_f1:.4f}")

    # Save the threshold search results for inspection.
    pd.DataFrame(threshold_results).to_csv(
        OUTPUT_DIR / "threshold_tuning.csv",
        index=False,
    )

    # Refit on the entire training period.
    # Keep the test period untouched until final evaluation.
    print("\nRefitting model on the full training period...")

    model = create_model(n_features)
    model.fit(X_train, y_train)

    # Final test-set evaluation.
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "decision_threshold": float(best_threshold),
        "validation_f1_score": float(validation_f1),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "sensor_features": int(X.shape[1]),
        "failure_rate": float(y.mean()),
        "test_failures": int((y_test == 1).sum()),
        "test_passes": int((y_test == 0).sum()),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }

    print("\n" + "=" * 50)
    print("FINAL TEST-SET PERFORMANCE")
    print("=" * 50)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Pass", "Fail"],
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            cm,
            index=["Actual Pass", "Actual Fail"],
            columns=["Predicted Pass", "Predicted Fail"],
        )
    )

    # --------------------------------------------------
    # 6. SAVE MODEL AND METRICS
    # --------------------------------------------------

    # Bundle the model and its threshold together so the
    # dashboard can use the same decision rule.
    model_bundle = {
        "model": model,
        "threshold": best_threshold,
        "feature_names": X.columns.tolist(),
        "positive_class": "Fail",
        "negative_class": "Pass",
    }

    joblib.dump(
        model_bundle,
        MODEL_DIR / "quality_model.joblib",
    )

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------
    # 7. SAVE TEST PREDICTIONS
    # --------------------------------------------------

    results = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "failure_probability": probabilities,
        "timestamp": timestamps.iloc[
            split_index:
        ].astype(str).to_numpy(),
    })

    results.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 8. SAVE CONFUSION MATRIX
    # --------------------------------------------------

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Pass", "Actual Fail"],
        columns=["Predicted Pass", "Predicted Fail"],
    )

    confusion_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    # --------------------------------------------------
    # 9. SAVE SELECTED FEATURE IMPORTANCE
    # --------------------------------------------------

    # Correctly map selected feature names through both
    # feature-removal stages to avoid the previous IndexError.
    imputer = model.named_steps["imputer"]
    variance_selector = model.named_steps["remove_constant"]
    feature_selector = model.named_steps["select_features"]

    # SimpleImputer can drop completely empty columns when
    # configured with its default feature behavior.
    imputed_names = X.columns[
        imputer.get_feature_names_out()
        .astype(str) == X.columns.to_numpy()
    ] if False else np.asarray(
        imputer.get_feature_names_out(),
        dtype=str,
    )

    remaining_names = imputed_names[
        variance_selector.get_support()
    ]

    selected_names = remaining_names[
        feature_selector.get_support()
    ]

    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    if len(selected_names) != len(importances):
        raise ValueError(
            "Feature importance length does not match "
            "the selected sensor names."
        )

    importance_df = pd.DataFrame({
        "sensor_feature": selected_names,
        "importance": importances,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTop 10 important sensor features:")
    print(importance_df.head(10).to_string(index=False))

    print("\nAll outputs saved successfully:")
    print(f"Model: {MODEL_DIR / 'quality_model.joblib'}")
    print(f"Metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Predictions: {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.csv'}")
    print(f"Feature importance: {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"Threshold tuning: {OUTPUT_DIR / 'threshold_tuning.csv'}")


if __name__ == "__main__":
    main()
