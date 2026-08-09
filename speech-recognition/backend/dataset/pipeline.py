"""
dataset/pipeline.py — Unified dataset pipeline and CLI orchestration.

Scans the raw datasets, parses filenames to extract labels and metadata,
normalizes emotion labels, splits into train/val/test, and outputs a 
unified metadata CSV.

Usage:
    python -m dataset.pipeline --datasets ravdess tess savee
"""

import os
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset.loaders import ravdess, tess, savee, crema_d, emo_db
from training.config import DEFAULT_CONFIG
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

# Map dataset names to their loader modules
LOADERS = {
    "ravdess": ravdess,
    "tess": tess,
    "savee": savee,
    "crema_d": crema_d,
    "emo_db": emo_db,
}


def create_metadata_csv(
    datasets: List[str],
    output_dir: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Run the full metadata extraction pipeline for the selected datasets.
    """
    raw_base_dir = Path(DEFAULT_CONFIG.dataset.raw_data_dir)
    all_records = []

    for ds_name in datasets:
        ds_name_lower = ds_name.lower().replace("-", "_")
        if ds_name_lower not in LOADERS:
            logger.error(f"Unknown dataset '{ds_name}'. Skipping.")
            continue
            
        loader_module = LOADERS[ds_name_lower]
        
        # Determine raw directory based on env vars or default path
        env_var_name = f"{ds_name_lower.upper().replace('_', '')}_PATH"
        if env_var_name == "CREMAD_PATH":
            env_var_name = "CREMA_D_PATH"
        elif env_var_name == "EMODB_PATH":
            env_var_name = "EMO_DB_PATH"
            
        env_path = os.getenv(env_var_name)
        if env_path:
            ds_dir = Path(env_path)
        else:
            # Fallback to dataset/raw/<DATASET_NAME>
            # Usually we expect the folder name to match the dataset name (case-insensitive)
            # Find the actual folder ignoring case
            matched_dirs = [d for d in raw_base_dir.iterdir() if d.is_dir() and d.name.lower() == ds_name.lower()]
            if matched_dirs:
                ds_dir = matched_dirs[0]
            else:
                ds_dir = raw_base_dir / ds_name.upper()

        logger.info(f"Processing dataset: {ds_name} from {ds_dir}")
        records = list(loader_module.parse_dataset(ds_dir))
        
        if not records:
            logger.warning(f"No valid records found for {ds_name} at {ds_dir}")
        else:
            logger.info(f"Loaded {len(records)} records from {ds_name}")
            all_records.extend(records)

    if not all_records:
        logger.error("No data found across any dataset. Check paths and data existence.")
        return None

    df = pd.DataFrame(all_records)
    
    # Generate unified splits (train/val/test)
    logger.info("Generating stratified Train/Val/Test splits...")
    
    # First split off the test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["emotion"]
    )
    
    # Then split the remaining into train and validation
    # Adjust validation size relative to the remaining data
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val["emotion"]
    )
    
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    
    logger.info(f"Data split: Train={len(train)} ({len(train)/len(df):.0%}), "
                f"Val={len(val)} ({len(val)/len(df):.0%}), "
                f"Test={len(test)} ({len(test)/len(df):.0%})")
                
    # Class distribution
    logger.info("\nClass Distribution:")
    dist = df["emotion"].value_counts()
    for em, count in dist.items():
        logger.info(f"  {em}: {count} ({count/len(df):.1%})")

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved unified metadata to {csv_path}")
    
    return df


@click.command()
@click.option(
    "--datasets", 
    "-d", 
    multiple=True,
    default=["ravdess", "tess"],
    help="Datasets to include (e.g., ravdess tess savee crema_d emo_db)"
)
@click.option(
    "--output", 
    "-o", 
    default=DEFAULT_CONFIG.dataset.processed_data_dir,
    help="Output directory for the metadata CSV"
)
@click.option(
    "--test-size", 
    default=DEFAULT_CONFIG.training.test_split,
    help="Fraction of data for test set"
)
@click.option(
    "--val-size", 
    default=DEFAULT_CONFIG.training.val_split,
    help="Fraction of data for validation set"
)
def main(datasets, output, test_size, val_size):
    """
    Speech Emotion Recognition — Dataset Processing Pipeline
    
    Scans raw dataset folders, normalizes labels, and generates a unified 
    metadata.csv with train/val/test splits.
    """
    setup_logger(log_level="INFO", log_file="logs/dataset_pipeline.log")
    
    # If the user passed a single string with spaces, split it
    if len(datasets) == 1 and " " in datasets[0]:
        datasets = datasets[0].split()
        
    logger.info(f"Starting dataset pipeline for: {', '.join(datasets)}")
    
    create_metadata_csv(
        datasets=datasets,
        output_dir=Path(output),
        test_size=test_size,
        val_size=val_size
    )


if __name__ == "__main__":
    main()
