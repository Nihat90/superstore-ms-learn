import argparse
import os

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()

    # Daten
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Pfad zur Superstore CSV-Datei",
    )

    # Random-Forest-Hyperparameter
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=1,
    )

    # Modell-Ausgabe
    parser.add_argument(
        "--model_output",
        type=str,
        default="outputs/model.joblib",
    )

    return parser.parse_args()


def prepare_data(df):
    # Datumsfelder konvertieren
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Feature Engineering
    df["Shipping Days"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

    df["Order Month"] = df["Order Date"].dt.month

    # Target
    df["Is Loss"] = (
        df["Profit"] < 0
    ).astype(int)

    return df


def main():
    args = parse_args()

    # -------------------------
    # 1. Daten laden
    # -------------------------
    print(f"Lade Daten aus: {args.data}")

    df = pd.read_csv(args.data)

    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # -------------------------
    # 2. Feature Engineering
    # -------------------------
    #df = prepare_data(df)

    # -------------------------
    # 3. Features + Target
    # -------------------------
    features = [
        "Sales",
        "Quantity",
        "Discount",
        "Category",
        "Sub-Category",
        "Segment",
        "Order Month",
        "Shipping Days",
    ]
    
    X = df[features]
    y = df["Is Loss"]

    # -------------------------
    # 4. Train/Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training: {X_train.shape}")
    print(f"Test:     {X_test.shape}")

    # -------------------------
    # 5. Preprocessing
    # -------------------------
    numeric_features = [
        "Sales",
        "Quantity",
        "Discount",
        "Order Month",
        "Shipping Days",
    ]

    categorical_features = [
        "Category",
        "Sub-Category",
        "Segment",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    # -------------------------
    # 6. Random Forest
    # -------------------------
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # -------------------------
    # 7. Training
    # -------------------------
    print("Training startet ...")

    model.fit(
        X_train,
        y_train,
    )

    # -------------------------
    # 8. Prediction
    # -------------------------
    predictions = model.predict(
        X_test
    )

    # -------------------------
    # 9. Evaluation
    # -------------------------
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    print("\nErgebnisse:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # -------------------------
    # 10. MLflow Logging
    # -------------------------
    mlflow.log_param(
        "n_estimators",
        args.n_estimators,
    )

    mlflow.log_param(
        "max_depth",
        args.max_depth,
    )

    mlflow.log_param(
        "min_samples_leaf",
        args.min_samples_leaf,
    )

    mlflow.log_metric(
        "accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "precision",
        precision,
    )

    mlflow.log_metric(
        "recall",
        recall,
    )

    mlflow.log_metric(
        "f1",
        f1,
    )

    # -------------------------
    # 11. Modell speichern
    # -------------------------
    model_directory = os.path.dirname(
        args.model_output
    )

    if model_directory:
        os.makedirs(
            model_directory,
            exist_ok=True,
        )

    joblib.dump(
        model,
        args.model_output,
    )

    print(
        f"\nModell gespeichert unter: "
        f"{args.model_output}"
    )


if __name__ == "__main__":
    main()
