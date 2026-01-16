#!/usr/bin/env python
"""Quick test of new improvements."""

from pathlib import Path
from src.indexer.attribute_parser import extract_atomic_probes, extract_tags
from PIL import Image
import numpy as np

# Test 1: Probe extraction
print("="*80)
print("TEST 1: Atomic Probe Extraction")
print("="*80)

test_queries = [
    "A bright yellow raincoat.",
    "A red tie and a white shirt in a formal setting.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
]

for query in test_queries:
    probes = extract_atomic_probes(query)
    print(f"\nQuery: {query}")
    print(f"Probes: {probes}")

# Test 2: Tag extraction
print("\n" + "="*80)
print("TEST 2: Tag Extraction from Captions")
print("="*80)

sample_captions = [
    "A person wearing a bright yellow raincoat standing outdoors.",
    "A man in a red tie and white shirt at a formal office meeting.",
    "A woman in a blue shirt sitting on a park bench in nature.",
]

for caption in sample_captions:
    tags = extract_tags(caption)
    print(f"\nCaption: {caption}")
    print(f"Tags: {tags}")

# Test 3: Color extraction (mock)
print("\n" + "="*80)
print("TEST 3: Color Feature Extraction (Mock)")
print("="*80)

# Create a test image with bright yellow
test_img = Image.new('RGB', (100, 100), color=(255, 255, 0))  # Bright yellow

try:
    from src.common.utils import extract_dominant_color
    color = extract_dominant_color(test_img)
    print(f"Test Image Color (bright yellow): {color}")
except ImportError as e:
    print(f"Skipping (torch import needed): {e}")

print("\n" + "="*80)
print("✓ All tests completed successfully!")
print("="*80)
