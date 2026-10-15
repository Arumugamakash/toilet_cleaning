"""
Dataset Clean & Split Pipeline
-------------------------------
Steps performed:
  1. Verify RGB <-> Depth pairing integrity
  2. Detect near-duplicate images (perceptual hashing)
  3. Detect invalid/bad depth frames (too many zero/missing pixels)
  4. Detect blurry RGB images (Laplacian variance)
  5. Write a cleaning report (what was removed and why)
  6. Split the REMAINING clean pairs into Train/Val/Test (70/20/10),
     keeping RGB+Depth pairs together.

Folder layout expected as input:
    dataset/
        rgb/   dataset_img_0001.png, dataset_img_0002.png, ...
        depth/ dataset_img_0001.png, dataset_img_0002.png, ...   (16-bit PNG, mm)

Output layout produced:
    dataset_split/
        train/rgb/...   train/depth/...
        val/rgb/...     val/depth/...
        test/rgb/...    test/depth/...
    cleaning_report.txt
"""

import os
import shutil
import random
import numpy as np
import cv2

# ============================================================
# CONFIG — edit these paths/thresholds for your project
# ============================================================
RGB_DIR = "/home/arz--1016/arm_basics/my_project/src/dataset/lid/rgb"
DEPTH_DIR = "/home/arz--1016/arm_basics/my_project/src/dataset/lid/depth"
OUTPUT_DIR = "/home/arz--1016/arm_basics/my_project/src/dataset/lid/dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

DUPLICATE_HASH_SIZE = 16         # average-hash grid size (16x16 = 256-bit hash)
                                  # Bigger hash = captures finer detail, so subtle
                                  # state differences (e.g. 10 deg vs 15 deg) are
                                  # NOT confused with true duplicates.
DUPLICATE_HAMMING_THRESHOLD = 4  # out of 256 bits (~1.5%) — very strict, only
                                  # catches near pixel-identical frames.
DUPLICATE_PIXEL_DIFF_THRESHOLD = 2.0   # mean absolute pixel difference (0-255 scale)
                                        # Second confirmation gate: even if hash looks
                                        # close, the actual pixels must also be almost
                                        # identical before we call it a duplicate.
DUPLICATE_SEARCH_WINDOW = 8       # only compare each image against the PREVIOUS N
                                   # files (by filename/capture order). True duplicates
                                   # happen back-to-back (camera hasn't moved yet) —
                                   # comparing against the whole dataset risks matching
                                   # two genuinely different states that just happen to
                                   # look statistically similar from far apart in time.
MAX_INVALID_DEPTH_RATIO = 0.35   # if >35% of depth pixels are 0/invalid, reject frame
BLUR_LAPLACIAN_THRESHOLD = 40.0  # below this variance => considered blurry

random.seed(42)


# ------------------------------------------------------------------
# 1. Perceptual hash (simple average-hash, no extra libraries needed)
# ------------------------------------------------------------------
def average_hash(image_path, hash_size=DUPLICATE_HASH_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    return bits


def hamming_distance(hash1, hash2):
    if hash1 is None or hash2 is None:
        return None
    return int(np.count_nonzero(hash1 != hash2))


def mean_pixel_diff(path1, path2, size=64):
    """Second confirmation check: actual pixel-level difference, not just hash."""      
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 999.0  # force "not a duplicate" if unreadable
    img1 = cv2.resize(img1, (size, size)).astype(np.float32)
    img2 = cv2.resize(img2, (size, size)).astype(np.float32)
    return float(np.mean(np.abs(img1 - img2)))


def find_duplicates(rgb_files, rgb_dir):
    """
    Returns a set of filenames considered TRUE duplicates of a recent file.

    A file is only flagged as duplicate if BOTH conditions hold against one
    of the last DUPLICATE_SEARCH_WINDOW files (by capture order):
      1. Perceptual hash is nearly identical (tight Hamming threshold)
      2. Actual pixel values are nearly identical (tight MAD threshold)

    This avoids accidentally removing images that are genuinely different
    lid states/angles/lighting but happen to look broadly similar.
    """
    hashes = {}          # fname -> hash
    order = []           # capture order, to limit comparison window
    duplicates = set()

    for fname in rgb_files:
        path = os.path.join(rgb_dir, fname)
        h = average_hash(path)

        # Only compare against the most recent N files (sliding window)
        candidates = order[-DUPLICATE_SEARCH_WINDOW:]

        is_dup = False
        for seen_fname in candidates:
            dist = hamming_distance(h, hashes[seen_fname])
            if dist is not None and dist <= DUPLICATE_HAMMING_THRESHOLD:
                # Hash says "close" -> confirm with actual pixel difference
                seen_path = os.path.join(rgb_dir, seen_fname)
                diff = mean_pixel_diff(path, seen_path)
                if diff <= DUPLICATE_PIXEL_DIFF_THRESHOLD:
                    duplicates.add(fname)
                    is_dup = True
                    break

        hashes[fname] = h
        order.append(fname)
        # NOTE: duplicates are still added to `order`/`hashes` so a run of
        # 3+ identical frames all get compared against the real original,
        # not just against each other.

    return duplicates


# ------------------------------------------------------------------
# 2. Invalid depth frame detection
# ------------------------------------------------------------------
def invalid_depth_ratio(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return 1.0  # treat unreadable as fully invalid
    total = depth.size
    invalid = np.count_nonzero(depth == 0)
    return invalid / total


# ------------------------------------------------------------------
# 3. Blur detection (on RGB)
# ------------------------------------------------------------------
def laplacian_variance(rgb_path):
    img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.lower().endswith(('.png', '.jpg'))])

    report_lines = []
    report_lines.append(f"Total RGB files found: {len(rgb_files)}")

    # ---- Step 1: Pairing check ----
    missing_depth = []
    valid_pairs = []
    for fname in rgb_files:
        depth_path = os.path.join(DEPTH_DIR, fname)
        if not os.path.exists(depth_path):
            missing_depth.append(fname)
        else:
            valid_pairs.append(fname)

    report_lines.append(f"Pairs missing depth file (excluded): {len(missing_depth)}")
    for f in missing_depth:
        report_lines.append(f"   MISSING DEPTH: {f}")

    # ---- Step 2: Duplicate detection (on the paired set) ----
    duplicates = find_duplicates(valid_pairs, RGB_DIR)
    report_lines.append(f"Near-duplicate images found (excluded): {len(duplicates)}")
    for f in sorted(duplicates):
        report_lines.append(f"   DUPLICATE: {f}")

    # ---- Step 3 & 4: Invalid depth + Blur check ----
    bad_depth = []
    blurry = []
    clean_files = []

    for fname in valid_pairs:
        if fname in duplicates:
            continue

        rgb_path = os.path.join(RGB_DIR, fname)
        depth_path = os.path.join(DEPTH_DIR, fname)

        inv_ratio = invalid_depth_ratio(depth_path)
        if inv_ratio > MAX_INVALID_DEPTH_RATIO:
            bad_depth.append((fname, inv_ratio))
            continue

        blur_score = laplacian_variance(rgb_path)
        if blur_score < BLUR_LAPLACIAN_THRESHOLD:
            blurry.append((fname, blur_score))
            continue

        clean_files.append(fname)

    report_lines.append(f"Rejected for bad/invalid depth (excluded): {len(bad_depth)}")
    for f, r in bad_depth:
        report_lines.append(f"   BAD DEPTH ({r*100:.1f}% invalid): {f}")

    report_lines.append(f"Rejected for blur (excluded): {len(blurry)}")
    for f, s in blurry:
        report_lines.append(f"   BLURRY (score={s:.1f}): {f}")

    report_lines.append(f"\nFINAL CLEAN DATASET SIZE: {len(clean_files)} pairs")

    # ---- Step 5: Split into Train/Val/Test ----
    random.shuffle(clean_files)
    n = len(clean_files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = clean_files[:n_train]
    val_files = clean_files[n_train:n_train + n_val]
    test_files = clean_files[n_train + n_val:]

    report_lines.append(f"\nSPLIT RESULT:")
    report_lines.append(f"   Train: {len(train_files)}")
    report_lines.append(f"   Val:   {len(val_files)}")
    report_lines.append(f"   Test:  {len(test_files)}")

    # ---- Copy files into split folders (keeping rgb/depth pairing) ----
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        rgb_out = os.path.join(OUTPUT_DIR, split_name, "rgb")
        depth_out = os.path.join(OUTPUT_DIR, split_name, "depth")
        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)
        for fname in files:
            shutil.copy2(os.path.join(RGB_DIR, fname), os.path.join(rgb_out, fname))
            shutil.copy2(os.path.join(DEPTH_DIR, fname), os.path.join(depth_out, fname))

    # ---- Write report ----
    with open(os.path.join(OUTPUT_DIR if os.path.isdir(OUTPUT_DIR) else ".", "cleaning_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nDone. Clean split dataset written to: {OUTPUT_DIR}/")
    print(f"Full report saved to: {OUTPUT_DIR}/cleaning_report.txt")


if __name__ == "__main__":
    main()