"""
Unit tests for document segmentation modules (Lines, Words, Characters).
"""

import numpy as np
import pytest
import cv2
from models.segmentation.line_detector import LineDetector, LineRegion
from models.segmentation.word_detector import WordDetector, WordRegion
from models.segmentation.char_segmentor import CharSegmentor, CharRegion


@pytest.fixture
def synthetic_document():
    """Create a synthetic binary document with 2 horizontal text lines, each having 2 words."""
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Line 1: y=[80..120]
    # Word 1: x=[50..200] (with 3 mock char blocks inside)
    cv2.rectangle(img, (50, 80), (90, 120), 255, -1)
    cv2.rectangle(img, (100, 80), (140, 120), 255, -1)
    cv2.rectangle(img, (150, 80), (190, 120), 255, -1)
    
    # Word 2: x=[300..450] (large horizontal gap of 110px separates words)
    cv2.rectangle(img, (300, 80), (340, 120), 255, -1)
    cv2.rectangle(img, (350, 80), (390, 120), 255, -1)
    cv2.rectangle(img, (400, 80), (440, 120), 255, -1)

    # Line 2: y=[250..290] (vertical gap of 130px separates lines)
    cv2.rectangle(img, (60, 250), (120, 290), 255, -1)
    cv2.rectangle(img, (280, 250), (360, 290), 255, -1)

    return img


def test_line_detector(synthetic_document):
    detector = LineDetector(min_line_height=15)
    lines = detector.detect(synthetic_document)
    
    # Expect 2 distinct horizontal lines detected
    assert len(lines) == 2
    assert all(isinstance(line, LineRegion) for line in lines)
    assert lines[0].y_min < lines[1].y_min  # Sorted top to bottom


def test_word_detector(synthetic_document):
    line_detector = LineDetector(min_line_height=15)
    lines = line_detector.detect(synthetic_document)

    word_detector = WordDetector()
    word_detector.detect_all_lines(synthetic_document, lines)
    
    # Line 1 should contain 2 detected words due to the 110px horizontal gap
    assert len(lines[0].words) == 2
    assert all(isinstance(word, WordRegion) for word in lines[0].words)
    assert lines[0].words[0].x_min < lines[0].words[1].x_min  # Sorted left to right


def test_char_segmentor(synthetic_document):
    line_detector = LineDetector(min_line_height=15)
    lines = line_detector.detect(synthetic_document)

    word_detector = WordDetector()
    word_detector.detect_all_lines(synthetic_document, lines)

    char_segmentor = CharSegmentor()
    char_segmentor.segment_all_words(synthetic_document, lines)
    
    # In Line 1, Word 1 has 3 distinct character bounding blocks
    words_line1 = lines[0].words
    assert len(words_line1[0].chars) == 3
    assert all(isinstance(ch, CharRegion) for ch in words_line1[0].chars)
    assert words_line1[0].chars[0].x_min < words_line1[0].chars[1].x_min
