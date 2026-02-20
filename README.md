# cryostructure-report

New Script: cryostructure_analysis.py - Analyzes permafrost core microcryostructures

Splits the 3-panel image (A, B, C)
1. Extracts texture features (intensity, gradients, entropy)
2. Detects ice lenses and layering patterns
3. Compares all three cryostructure types

## Outputs Generated:
Comprehensive visualization - 5-row analysis showing original images, grayscale, edge detection, ice lens detection, and color distributions
Feature comparison plots - 6 quantitative metrics compared across panels
CSV data - All extracted features for further analysis
Text report - Detailed analysis with interpretations

## Key Findings:
Panel A (Micro-layered lenticular): Most complex structure, 45 ice lenses
Panel B (Micro-lenticular): Strongest layering, 83 ice lenses detected
Panel C (Micro-suspended): Less layered (0.77 ratio), 22 ice lenses
