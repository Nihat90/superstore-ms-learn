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
