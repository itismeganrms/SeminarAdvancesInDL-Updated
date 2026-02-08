import os
import random
import pandas as pd
from pathlib import Path
import click


@click.command()
@click.option(
    "--root_dir", required=True, help="Path to dataset root (contains patient folders)."
)
@click.option("--csv_path", required=True, help="Path to the data.csv file.")
@click.option("--output_dir", required=True, help="Where to save the split CSV files.")
@click.option("--seed", default=42, help="Random seed.")
@click.option(
    "--val_ratio", default=0.15, help="Fraction of data to use for validation."
)
@click.option("--test_ratio", default=0.15, help="Fraction of data to use for test.")
def main(root_dir, csv_path, output_dir, seed, val_ratio, test_ratio):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=None, engine="python")
    patient_ids = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]

    df_filtered = df[df["Patient"].isin(patient_ids)].copy()
    df_filtered = df_filtered.sample(frac=1, random_state=seed)  # Shuffle

    total = len(df_filtered)
    n_val = int(val_ratio * total)
    n_test = int(test_ratio * total)
    n_train = total - n_val - n_test

    df_train = df_filtered.iloc[:n_train]
    df_val = df_filtered.iloc[n_train : n_train + n_val]
    df_test = df_filtered.iloc[n_train + n_val :]

    df_train.to_csv(output_dir / "train_split.csv", index=False)
    df_val.to_csv(output_dir / "val_split.csv", index=False)
    df_test.to_csv(output_dir / "test_split.csv", index=False)

    print(
        f"✅ Done! Saved {len(df_train)} train / {len(df_val)} val / {len(df_test)} test samples to {output_dir}"
    )


if __name__ == "__main__":
    main()
