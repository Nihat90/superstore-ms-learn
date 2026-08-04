import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Pfad zur Superstore CSV-Datei",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output-Ordner fuer die vorbereiteten Daten",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading raw data from: {args.data}")

    df = pd.read_csv(
        args.data,
        encoding="latin-1"
    )

    print(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Datumsfelder
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

        # Nur die Features behalten, die unser Modell braucht
    columns = [
        "Sales",
        "Quantity",
        "Discount",
        "Category",
        "Sub-Category",
        "Segment",
        "Order Month",
        "Shipping Days",
        "Is Loss",
    ]

    prepared_df = df[columns]

    # Output-Ordner erstellen
    os.makedirs(
        args.output,
        exist_ok=True
    )

    # Dateipfad erzeugen
    output_file = os.path.join(
        args.output,
        "prepared.csv"
    )

    # CSV speichern
    prepared_df.to_csv(
        output_file,
        index=False
    )

    print(f"Prepared dataset saved to: {output_file}")
    print(f"Prepared shape: {prepared_df.shape}")
if __name__ == "__main__":
    main()
