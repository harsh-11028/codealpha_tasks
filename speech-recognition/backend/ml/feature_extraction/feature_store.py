"""
ml/feature_extraction/feature_store.py — HDF5 Feature Store.

Manages efficient saving and loading of extracted feature arrays for training.
Using HDF5 allows us to stream massive feature arrays directly from disk 
without loading the entire dataset into RAM.
"""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Interface for HDF5-based feature storage.
    
    Structure inside HDF5:
    /features/[file_id] -> dataset of shape (n_features, time_frames)
    /labels/[file_id]   -> string emotion label
    /metadata/[file_id] -> JSON string of additional metadata
    """

    def __init__(self, store_path: Union[str, Path], mode: str = "a"):
        """
        Initialize the feature store.
        
        Args:
            store_path: Path to the .h5 file.
            mode: 'a' for read/write/create, 'r' for read-only, 'w' to overwrite.
        """
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        
        # Ensure base groups exist if creating/appending
        if self.mode in ('a', 'w'):
            with h5py.File(self.store_path, self.mode) as f:
                if "features" not in f:
                    f.create_group("features")
                if "labels" not in f:
                    f.create_group("labels")
                if "metadata" not in f:
                    f.create_group("metadata")
                    
        logger.debug(f"FeatureStore initialized at {self.store_path} (mode={mode})")

    def save_feature(self, file_id: str, features: np.ndarray, label: str, metadata: str = "") -> None:
        """Save a single file's extracted features and label."""
        with h5py.File(self.store_path, "a") as f:
            # Overwrite if exists
            if file_id in f["features"]:
                del f["features"][file_id]
            if file_id in f["labels"]:
                del f["labels"][file_id]
            if file_id in f["metadata"]:
                del f["metadata"][file_id]

            # Compress features using gzip (level 4 is a good speed/size tradeoff)
            f["features"].create_dataset(
                file_id, 
                data=features, 
                compression="gzip", 
                compression_opts=4
            )
            f["labels"].create_dataset(file_id, data=label)
            f["metadata"].create_dataset(file_id, data=metadata)

    def save_batch(self, batch_dict: dict[str, dict]) -> None:
        """
        Save a batch of features efficiently.
        
        Args:
            batch_dict: { file_id: {"features": ndarray, "label": str, "metadata": str} }
        """
        with h5py.File(self.store_path, "a") as f:
            for file_id, data in batch_dict.items():
                if file_id in f["features"]:
                    del f["features"][file_id]
                if file_id in f["labels"]:
                    del f["labels"][file_id]
                if file_id in f["metadata"]:
                    del f["metadata"][file_id]
                    
                f["features"].create_dataset(
                    file_id, 
                    data=data["features"], 
                    compression="gzip", 
                    compression_opts=4
                )
                f["labels"].create_dataset(file_id, data=data["label"])
                f["metadata"].create_dataset(file_id, data=data.get("metadata", ""))
                
        logger.info(f"Saved {len(batch_dict)} features to {self.store_path}")

    def get_feature(self, file_id: str) -> Optional[np.ndarray]:
        """Retrieve features for a single file."""
        with h5py.File(self.store_path, "r") as f:
            if file_id in f["features"]:
                return f["features"][file_id][:]
            return None

    def get_label(self, file_id: str) -> Optional[str]:
        """Retrieve label for a single file."""
        with h5py.File(self.store_path, "r") as f:
            if file_id in f["labels"]:
                return f["labels"][file_id][()].decode("utf-8")
            return None

    def list_file_ids(self) -> list[str]:
        """Get a list of all saved file IDs."""
        with h5py.File(self.store_path, "r") as f:
            return list(f["features"].keys())

    def get_stats(self) -> dict:
        """Return basic statistics about the stored data."""
        if not self.store_path.exists():
            return {"count": 0, "size_mb": 0.0}
            
        with h5py.File(self.store_path, "r") as f:
            count = len(f["features"])
            
            # Get shape of first item if exists
            shape = "unknown"
            if count > 0:
                first_key = list(f["features"].keys())[0]
                shape = str(f["features"][first_key].shape)
                
        size_mb = self.store_path.stat().st_size / (1024 * 1024)
        
        return {
            "count": count,
            "feature_shape": shape,
            "size_mb": round(size_mb, 2)
        }
