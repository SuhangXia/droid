from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def compare_reference_frames(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    candidate = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or candidate is None:
        return {
            "image_similarity": None,
            "estimated_translation": None,
            "estimated_rotation": None,
            "homography_confidence": None,
            "consistency_class": "unknown",
        }
    candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]))
    ref_small = cv2.resize(reference, (320, 240)).astype(np.float32)
    can_small = cv2.resize(candidate, (320, 240)).astype(np.float32)
    shift, response = cv2.phaseCorrelate(ref_small, can_small)
    translation = float(np.hypot(*shift))
    hist_ref = cv2.calcHist([reference], [0], None, [64], [0, 256])
    hist_can = cv2.calcHist([candidate], [0], None, [64], [0, 256])
    histogram_similarity = float(cv2.compareHist(hist_ref, hist_can, cv2.HISTCMP_CORREL))

    detector = cv2.ORB_create(nfeatures=700)
    key_ref, desc_ref = detector.detectAndCompute(reference, None)
    key_can, desc_can = detector.detectAndCompute(candidate, None)
    rotation = None
    homography_confidence = 0.0
    if desc_ref is not None and desc_can is not None and len(key_ref) >= 8 and len(key_can) >= 8:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_ref, desc_can, k=2)
        good = [left for left, right in matches if left.distance < 0.75 * right.distance]
        if len(good) >= 8:
            source = np.float32([key_ref[item.queryIdx].pt for item in good])
            target = np.float32([key_can[item.trainIdx].pt for item in good])
            homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
            if homography is not None and mask is not None:
                homography_confidence = float(mask.mean())
                rotation = float(np.degrees(np.arctan2(homography[1, 0], homography[0, 0])))
    combined_similarity = float(
        np.clip(
            0.55 * max(0.0, histogram_similarity) + 0.25 * max(0.0, response) + 0.20 * homography_confidence, 0.0, 1.0
        )
    )
    consistency_class = (
        "standard"
        if combined_similarity >= 0.70 and translation < 12.0
        else "minor_shift"
        if combined_similarity >= 0.48 and translation < 35.0
        else "major_shift"
    )
    return {
        "image_similarity": round(combined_similarity, 4),
        "estimated_translation": round(translation, 4),
        "estimated_rotation": None if rotation is None else round(rotation, 4),
        "homography_confidence": round(homography_confidence, 4),
        "consistency_class": consistency_class,
    }
