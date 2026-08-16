"""
Connected component analysis (CCA) for OCR feature extraction.

Connected component analysis identifies individual blobs of connected
foreground pixels. Beyond segmentation, CCA provides rich shape
descriptors that characterize character structure:

  - Area, perimeter, centroid
  - Bounding box dimensions and aspect ratio
  - Extent (foreground fraction of bounding box)
  - Solidity (area / convex hull area)
  - Euler number (topology: holes count)
  - Hu moments (shape invariants)
  - Zernike moments (rotation-invariant shape descriptors)

These features complement CNN features and are useful for:
  - Pre-classification filtering (noise vs. character)
  - Character disambiguation (e.g., 'l' vs 'I' vs '1')
  - Hybrid model inputs alongside HOG features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.measure import regionprops, label as skimage_label
from skimage.morphology import convex_hull_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component descriptor
# ---------------------------------------------------------------------------

@dataclass
class ComponentDescriptor:
    """
    Shape descriptors for a single connected component.

    Attributes:
        label_id:      Component label (from cv2.connectedComponentsWithStats).
        area:          Number of foreground pixels.
        perimeter:     Perimeter length (OpenCV contour arclength).
        centroid:      (cx, cy) centroid in image coordinates.
        bbox:          (x, y, w, h) bounding box.
        aspect_ratio:  Width / Height.
        extent:        area / bounding_box_area.
        solidity:      area / convex_hull_area (1.0 = convex shape).
        euler_number:  Connectivity topology number.
        hu_moments:    7-element array of Hu moment invariants.
        mean_intensity: Mean pixel intensity within component.
    """
    label_id: int
    area: int
    perimeter: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    aspect_ratio: float
    extent: float
    solidity: float
    euler_number: int
    hu_moments: np.ndarray
    mean_intensity: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """
        Flatten all descriptors into a single float feature vector.

        Returns:
            Float64 array of length 14 + 7 = 21 features.
        """
        scalar_features = np.array([
            self.area,
            self.perimeter,
            self.centroid[0],
            self.centroid[1],
            self.bbox[2],          # width
            self.bbox[3],          # height
            self.aspect_ratio,
            self.extent,
            self.solidity,
            float(self.euler_number),
            self.mean_intensity,
        ], dtype=np.float64)
        return np.concatenate([scalar_features, self.hu_moments])

    def to_dict(self) -> dict:
        """Serialize for API / logging."""
        return {
            "label_id": self.label_id,
            "area": self.area,
            "perimeter": round(self.perimeter, 2),
            "centroid": {"cx": round(self.centroid[0], 2), "cy": round(self.centroid[1], 2)},
            "bbox": {"x": self.bbox[0], "y": self.bbox[1], "w": self.bbox[2], "h": self.bbox[3]},
            "aspect_ratio": round(self.aspect_ratio, 4),
            "extent": round(self.extent, 4),
            "solidity": round(self.solidity, 4),
            "euler_number": self.euler_number,
            "hu_moments": self.hu_moments.tolist(),
            "mean_intensity": round(self.mean_intensity, 4),
        }


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def compute_hu_moments(binary_component: np.ndarray) -> np.ndarray:
    """
    Compute the 7 Hu moment invariants for a binary component.

    Hu moments are invariant to translation, scale, and rotation —
    powerful shape descriptors for character recognition.

    Args:
        binary_component: Binary uint8 mask of one component.

    Returns:
        Float64 array of 7 log-scaled Hu moments.
    """
    moments = cv2.moments(binary_component)
    hu = cv2.HuMoments(moments).flatten()
    # Log scale to equalize the large dynamic range
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def compute_solidity(binary_mask: np.ndarray) -> float:
    """
    Compute solidity = area / convex_hull_area.

    Args:
        binary_mask: Binary uint8 mask for a single component.

    Returns:
        Solidity in [0, 1]. Values close to 1 indicate convex shapes.
    """
    area = binary_mask.sum()
    if area == 0:
        return 0.0
    try:
        hull = convex_hull_image(binary_mask > 0)
        hull_area = hull.sum()
        return float(area / max(hull_area, 1))
    except Exception:
        return 1.0


def compute_euler_number(binary_mask: np.ndarray) -> int:
    """
    Compute Euler number (number of objects - number of holes).

    Characters like 'O', 'B', 'g' have holes (euler < 1),
    while 'I', 'T', 'L' do not (euler = 1).

    Args:
        binary_mask: Binary uint8 image.

    Returns:
        Euler number (integer).
    """
    try:
        labeled = skimage_label(binary_mask > 0)
        props = regionprops(labeled)
        if props:
            return int(props[0].euler_number)
        return 1
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Main CCA class
# ---------------------------------------------------------------------------

class ConnectedComponentAnalyzer:
    """
    Performs connected component analysis and extracts shape descriptors.

    Usage:
        analyzer = ConnectedComponentAnalyzer()

        # Get all component descriptors
        descriptors = analyzer.analyze(binary_image)

        # Get feature matrix for ML
        feature_matrix = analyzer.get_feature_matrix(binary_image)
    """

    def __init__(
        self,
        min_area: int = 10,
        max_area_ratio: float = 0.95,
        connectivity: int = 8,
        compute_solidity: bool = True,
        compute_euler: bool = True,
    ) -> None:
        """
        Args:
            min_area:         Minimum component area in pixels.
            max_area_ratio:   Max fraction of total image area for a component.
            connectivity:     8-connectivity (default) or 4-connectivity.
            compute_solidity: Compute convex hull solidity (slightly slower).
            compute_euler:    Compute Euler number (requires skimage).
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.connectivity = connectivity
        self._compute_solidity = compute_solidity
        self._compute_euler = compute_euler

    def analyze(
        self,
        binary_image: np.ndarray,
    ) -> List[ComponentDescriptor]:
        """
        Analyze connected components in a binary image.

        Args:
            binary_image: Binary uint8 image (text=255, background=0).

        Returns:
            List of ComponentDescriptor objects sorted by area (descending).
        """
        if binary_image.dtype != np.uint8:
            binary_image = (binary_image * 255).astype(np.uint8)

        h, w = binary_image.shape
        total_area = h * w

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=self.connectivity
        )

        descriptors: List[ComponentDescriptor] = []

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            if area > total_area * self.max_area_ratio:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            cw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            ch = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])
            aspect_ratio = cw / max(ch, 1)
            bbox_area = cw * ch
            extent = area / max(bbox_area, 1)

            # Isolated component mask
            component_mask = (labels == label_id).astype(np.uint8) * 255
            region_mask = component_mask[y:y + ch, x:x + cw]

            # Perimeter via contour
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours) if contours else 0.0

            # Hu moments
            hu = compute_hu_moments(region_mask)

            # Solidity
            solidity = (
                compute_solidity(region_mask) if self._compute_solidity else 1.0
            )

            # Euler number
            euler = (
                compute_euler_number(region_mask) if self._compute_euler else 1
            )

            # Mean intensity within bounding box on original image
            roi = binary_image[y:y + ch, x:x + cw]
            mean_intensity = float(roi.mean()) / 255.0

            descriptors.append(ComponentDescriptor(
                label_id=label_id,
                area=area,
                perimeter=perimeter,
                centroid=(cx, cy),
                bbox=(x, y, cw, ch),
                aspect_ratio=aspect_ratio,
                extent=extent,
                solidity=solidity,
                euler_number=euler,
                hu_moments=hu,
                mean_intensity=mean_intensity,
            ))

        # Sort by area descending (largest component first)
        descriptors.sort(key=lambda d: d.area, reverse=True)
        logger.debug("CCA: %d components analyzed.", len(descriptors))
        return descriptors

    def get_feature_matrix(
        self,
        binary_image: np.ndarray,
    ) -> np.ndarray:
        """
        Get a matrix of shape (N, 21) where N = number of valid components.

        Each row is the feature vector of one component.
        Returns empty (0, 21) array if no components found.

        Args:
            binary_image: Binary uint8 image.

        Returns:
            Float64 ndarray of shape (N, 21).
        """
        descriptors = self.analyze(binary_image)
        if not descriptors:
            return np.zeros((0, 21), dtype=np.float64)
        return np.stack([d.to_feature_vector() for d in descriptors])

    def get_aggregate_features(
        self,
        binary_image: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate per-component features into a single image-level vector.

        Computes (mean, std, min, max) over each feature dimension across
        all components, then concatenates — gives a fixed-length representation
        regardless of component count.

        Args:
            binary_image: Binary uint8 image.

        Returns:
            Float64 array of length 21 * 4 = 84.
        """
        matrix = self.get_feature_matrix(binary_image)
        if matrix.shape[0] == 0:
            return np.zeros(21 * 4, dtype=np.float64)
        return np.concatenate([
            matrix.mean(axis=0),
            matrix.std(axis=0),
            matrix.min(axis=0),
            matrix.max(axis=0),
        ])

    def visualize(
        self,
        binary_image: np.ndarray,
        descriptors: List[ComponentDescriptor],
        show_centroids: bool = True,
        show_bboxes: bool = True,
        show_ids: bool = True,
    ) -> np.ndarray:
        """
        Visualize connected components with color-coding.

        Args:
            binary_image: Binary uint8 image.
            descriptors:  Analyzed component descriptors.
            show_centroids: Draw centroid dots.
            show_bboxes:    Draw bounding boxes.
            show_ids:       Label each component with its area.

        Returns:
            BGR image with visualizations.
        """
        vis = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        # Color-code by area (large=blue, small=red)
        max_area = max((d.area for d in descriptors), default=1)
        for d in descriptors:
            norm = d.area / max_area
            color = (int(255 * (1 - norm)), 100, int(255 * norm))  # blue→red

            if show_bboxes:
                x, y, bw, bh = d.bbox
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 1)

            if show_centroids:
                cx, cy = int(d.centroid[0]), int(d.centroid[1])
                cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

            if show_ids:
                x, y = d.bbox[0], d.bbox[1]
                cv2.putText(
                    vis, f"{d.area}", (x, max(y - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
                )

        return vis

    @staticmethod
    def count_holes(binary_image: np.ndarray) -> int:
        """
        Count the number of holes (enclosed background regions) in the image.

        Useful for disambiguating characters like 'O' (1 hole) vs 'B' (2 holes)
        vs 'H' (0 holes).

        Args:
            binary_image: Binary uint8 image (text=255).

        Returns:
            Number of holes (background connected components that don't
            touch the image border).
        """
        inverted = cv2.bitwise_not(binary_image)
        # Flood fill from border to mark external background
        h, w = inverted.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flooded = inverted.copy()
        cv2.floodFill(flooded, mask, (0, 0), 128)

        # Remaining white pixels in inverted = holes
        holes = (flooded == 255).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(holes, connectivity=8)
        return max(0, num_labels - 1)
