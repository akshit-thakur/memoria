"""Blur detection scoring module — 6-model weighted ensemble.

In plain English: this package computes a single "how blurry is this
photo?" score (0 = razor sharp, 1 = uniform mush) by combining six
traditional computer-vision detectors (Laplacian variance, Tenengrad
gradient, CPBD probability-of-blur, FFT high-frequency energy, Brenner
gradient, local standard deviation) into one weighted average.

Public entry points:
  • ``score_image(path)``          — single float score.
  • ``score_image_detailed(path)`` — score plus per-detector breakdown.
  • ``classify(score)``            — bucket the score into a label.
  • ``BLUR_THRESHOLD``             — default cutoff (overridable via
                                     ``MEMORIA_BLUR_THRESHOLD``).
"""

from .blur_detect import score_image, score_image_detailed, classify, BLUR_THRESHOLD
