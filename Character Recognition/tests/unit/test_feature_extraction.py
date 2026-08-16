"""
Unit tests for feature extraction modules (HOG, Edge detection, CCA).
"""

import numpy as np
import pytest
import cv2
from models.feature_extraction.hog_extractor import HOGExtractor
from models.feature_extraction.edge_detector import EdgeDetector, detect_canny_edges
from models.feature_extraction.connected_components import ConnectedComponentAnalyzer


@pytest.fixture
def char_image():
    """Synthetic character image (letter 'T' silhouette)."""
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(img, (12, 12), (52, 22), 255, -1)   # Top bar
    cv2.rectangle(img, (27, 22), (37, 52), 255, -1)   # Vertical stem
    return img


def test_hog_feature_extractor(char_image):
    extractor = HOGExtractor(image_size=(64, 64))
    features, _ = extractor.extract(char_image)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    assert len(features) > 0


def test_edge_features(char_image):
    edges = detect_canny_edges(char_image)
    assert edges.shape == (64, 64)
    
    detector = EdgeDetector(method="canny", image_size=(64, 64))
    features = detector.extract_features(char_image)
    assert isinstance(features, np.ndarray)
    assert len(features) > 0


def test_connected_components(char_image):
    # Add a tiny speck of noise (2x2 pixel block)
    char_image[2, 2:4] = 255
    char_image[3, 2:4] = 255
    
    analyzer_all = ConnectedComponentAnalyzer(min_area=1)
    components = analyzer_all.analyze(char_image)
    # Should detect at least 2 components: main 'T' silhouette and the tiny speck
    assert len(components) >= 2

    # Filter out small noise speck by configuring min_area=20
    analyzer_filtered = ConnectedComponentAnalyzer(min_area=20)
    new_components = analyzer_filtered.analyze(char_image)
    
    # After filtering by min_area=20, only the large 'T' silhouette should remain
    assert len(new_components) < len(components)
    assert all(c.area >= 20 for c in new_components)
