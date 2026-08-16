"""
Unit tests for image preprocessing module.
Tests denoising, contrast enhancement (CLAHE), binarization, and morphological cleanup.
"""

import numpy as np
import pytest
import cv2
from models.preprocessing.image_processor import (
    to_grayscale,
    remove_noise,
    enhance_contrast_clahe,
    binarize,
    morphological_close,
)
from models.preprocessing.deskew import deskew_image


@pytest.fixture
def sample_image():
    """Create a synthetic RGB noisy text image."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 240
    # Draw some mock dark text stripes
    cv2.rectangle(img, (50, 80), (350, 120), (20, 20, 20), -1)
    # Add random noise
    noise = np.random.randint(0, 50, (200, 400, 3), dtype=np.uint8)
    return cv2.subtract(img, noise)


def test_to_grayscale(sample_image):
    gray = to_grayscale(sample_image)
    assert gray.ndim == 2
    assert gray.shape == (200, 400)
    assert gray.dtype == np.uint8


def test_remove_noise(sample_image):
    gray = to_grayscale(sample_image)
    denoised = remove_noise(gray, kernel_size=3)
    assert denoised.shape == gray.shape
    # Denoised image should have smoother local variance than original noisy gray
    assert np.var(denoised) <= np.var(gray) + 1.0


def test_enhance_contrast_clahe(sample_image):
    gray = to_grayscale(sample_image)
    enhanced = enhance_contrast_clahe(gray, clip_limit=2.0, tile_grid=(8, 8))
    assert enhanced.shape == gray.shape
    assert enhanced.dtype == np.uint8


def test_binarize(sample_image):
    gray = to_grayscale(sample_image)
    binary = binarize(gray, method="otsu")
    assert binary.shape == gray.shape
    # Output must be binary (only 0 and 255 values)
    unique_vals = np.unique(binary)
    assert all(val in (0, 255) for val in unique_vals)


def test_morphological_close(sample_image):
    gray = to_grayscale(sample_image)
    binary = binarize(gray, method="otsu")
    closed = morphological_close(binary, kernel_size=3)
    assert closed.shape == binary.shape
    assert closed.dtype == np.uint8


def test_deskew_image():
    """Test deskewing on a slightly rotated horizontal line."""
    img = np.zeros((300, 300), dtype=np.uint8)
    # Draw horizontal text bar and rotate by 5 degrees
    cv2.rectangle(img, (50, 140), (250, 160), 255, -1)
    M = cv2.getRotationMatrix2D((150, 150), 5, 1.0)
    rotated = cv2.warpAffine(img, M, (300, 300))
    
    deskewed = deskew_image(rotated, max_angle=15.0)
    assert deskewed.shape == (300, 300)
    assert deskewed.dtype == np.uint8
