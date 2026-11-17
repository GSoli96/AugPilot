from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Configuration
INPUT_DIR = Path("imbalance_datasets_SMOTE")
OUTPUT_DIR = Path("smote_augmented_samples")
TARGET_COLUMN = "class"
TEST_SIZE = 0.30
RANDOM_STATE = 42


def augment_dataset(csv_path: Path) -> None:
    """Apply SMOTE to the training split and save only the synthetic samples."""
    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {csv_path.name}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split first: SMOTE must be applied only to the training data.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Generate synthetic minority-class samples.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # SMOTE keeps the original training rows first; the additional rows
    # at the end are the synthetic samples.
    n_original = len(X_train)
    X_synthetic = X_resampled.iloc[n_original:] if hasattr(
        X_resampled, "iloc"
    ) else pd.DataFrame(X_resampled[n_original:], columns=X.columns)
    y_synthetic = y_resampled.iloc[n_original:] if hasattr(
        y_resampled, "iloc"
    ) else pd.Series(y_resampled[n_original:], name=TARGET_COLUMN)

    synthetic_df = X_synthetic.copy()
    synthetic_df[TARGET_COLUMN] = y_synthetic.to_numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{csv_path.stem}_synthetic_smote.csv"
    synthetic_df.to_csv(output_path, index=False)

    print(f"Dataset: {csv_path.name}")
    print(f"Training samples before SMOTE: {n_original}")
    print(f"Training samples after SMOTE:  {len(X_resampled)}")
    print(f"Synthetic samples generated:   {len(synthetic_df)}")
    print(f"Saved to: {output_path}")


def main() -> None:
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {INPUT_DIR}")
        return

    for csv_path in csv_files:
        augment_dataset(csv_path)


if __name__ == "__main__":
    main()
