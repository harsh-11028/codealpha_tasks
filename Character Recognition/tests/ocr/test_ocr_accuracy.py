"""
Unit tests for OCR text reconstruction and multi-engine merging logic.
Verifies spatial reading order sorting, word gap calculation, and output cleanup.
"""

import pytest
from models.ocr.text_reconstructor import (
    sort_chars_reading_order,
    compute_word_gaps,
    detect_word_boundaries,
    TextReconstructor,
)

class MockBox:
    """Mock character bounding box region."""
    def __init__(self, char: str, x_min: int, x_max: int, y_min: int, y_max: int, conf: float = 0.95):
        self.char = char
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.height = y_max - y_min
        self.confidence = conf
        self.width = x_max - x_min


def test_spatial_reading_order():
    """Verify out-of-order character regions get correctly sorted left-to-right, top-to-bottom."""
    # Line 1 (y ≈ 10..30): 'H', 'I'
    c_i = MockBox('I', 40, 55, 12, 32)
    c_h = MockBox('H', 10, 30, 10, 30)
    
    # Line 2 (y ≈ 60..80): 'O', 'C', 'R'
    c_r = MockBox('R', 80, 100, 62, 82)
    c_c = MockBox('C', 45, 65, 60, 80)
    c_o = MockBox('O', 10, 32, 61, 81)

    # Pass in shuffled order
    shuffled = [c_r, c_i, c_o, c_h, c_c]
    sorted_chars = sort_chars_reading_order(shuffled, line_height_tolerance=0.5)

    extracted = "".join([b.char for b in sorted_chars])
    assert extracted == "HIOCR"


def test_word_gap_boundary_detection():
    """Verify large horizontal gaps between characters trigger word splitting."""
    # "A B": character 'A' from 10..30, large gap of 60px, character 'B' from 90..110
    c_a = MockBox('A', 10, 30, 10, 30)
    c_b = MockBox('B', 90, 110, 10, 30)
    c_c = MockBox('C', 115, 135, 10, 30) # normal small gap of 5px after B

    chars = [c_a, c_b, c_c]
    gaps = compute_word_gaps(chars)
    assert gaps == [60.0, 5.0]

    boundaries = detect_word_boundaries(chars, multiplier=2.0)
    # Index 1 (before 'B') should be flagged as a word boundary
    assert 1 in boundaries


def test_reconstruct_from_chars():
    reconstructor = TextReconstructor(word_gap_multiplier=2.5, apply_cleanup=True)
    # Simulate "AI OCR": 'A','I', large gap, 'O','C','R'
    chars = [
        MockBox('A', 10, 30, 10, 30, 0.99),
        MockBox('I', 35, 50, 10, 30, 0.98),
        MockBox('O', 120, 140, 10, 30, 0.95),
        MockBox('C', 145, 165, 10, 30, 0.96),
        MockBox('R', 170, 190, 10, 30, 0.94),
    ]

    text, mean_conf = reconstructor.reconstruct_from_chars(chars)
    assert text == "AI OCR"
    assert mean_conf > 0.95


def test_merge_engine_outputs():
    reconstructor = TextReconstructor()
    merged_text, conf, engine = reconstructor.merge_engine_outputs(
        custom_text="Handwritten Text", custom_conf=0.96,
        easyocr_text="Handwritten Text", easyocr_conf=0.88,
        tesseract_text="Handvritten Txt", tesseract_conf=0.72,
        strategy="best_confidence"
    )
    assert merged_text == "Handwritten Text"
    assert engine == "custom"
    assert conf == 0.96
