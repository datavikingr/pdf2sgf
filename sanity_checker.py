import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import argparse

def test_whiteness(image_path: Path, threshold=0.01, visualize=False): #FAILED HEURISTIC
    print(f"🔍 Testing: {image_path.name}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Heuristic: count how much is *not* close to white
    non_white_ratio = np.mean(gray < 250)
    percent = non_white_ratio * 100

    if non_white_ratio < threshold:
        print(f"⚠️ Skipping — only {percent:.2f}% ink (below {threshold*100:.2f}%)")
    else:
        print(f"✅ Keep — {percent:.2f}% ink")

    if visualize:
        cv2.imshow(f"{image_path.name} ({percent:.2f}% ink)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_grid_structure(image_path: Path, hough_thresh=120, visualize=False):
    print(f"🔍 Testing: {image_path.name}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)

    num_lines = 0 if lines is None else len(lines)
    print(f"📏 Detected lines: {num_lines}")

    if visualize:
        display = img.copy()
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow(f"{image_path.name} - {num_lines} lines", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_column_count(image_path: Path, show_plot=False) -> int:
    print(f"🔍 Checking columns in {image_path.name}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("⚠️ Could not read image.")
        return 0

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Close small vertical gaps to merge text blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 100))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Sum black pixels vertically
    col_sums = np.sum(closed > 0, axis=0)

    # Normalize and smooth
    smoothed = cv2.GaussianBlur(col_sums.astype(np.float32), (51, 1), 0)

    # Threshold to find active columns
    threshold = np.max(smoothed) * 0.5
    active_cols = smoothed > threshold

    # Count contiguous blocks wider than MIN_COL_WIDTH
    MIN_COL_WIDTH = 100  # tweak as needed
    col_count = 0
    width = 0
    in_col = False

    for is_active in active_cols:
        if is_active:
            width += 1
            in_col = True
        elif in_col:
            if width >= MIN_COL_WIDTH:
                col_count += 1
            width = 0
            in_col = False
    # Edge case: end in column
    if in_col and width >= MIN_COL_WIDTH:
        col_count += 1

    print(f"✅ Detected columns: {col_count}")

    col_count = max(1, min(col_count, 3))  # Clamp between 1 and 3

    print(f"Clamped columns: {col_count}")

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(smoothed, label="Smoothed Column Sum")
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f"{image_path.name} – Detected Columns: {col_count}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return col_count

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Test image for mostly-white heuristic.")
    #parser.add_argument("image", type=str, help="Path to image file")
    #parser.add_argument("--threshold", type=float, default=0.01, help="Minimum ink ratio to keep (0.01 = 1%)")
    #parser.add_argument("--visualize", action="store_true", help="Show the image while reporting ink ratio")
    #args = parser.parse_args()
    #test_whiteness(Path(args.image), threshold=args.threshold, visualize=args.visualize)
    #NOTE: test_whiteness continued the same false-positive on problem129, so that isn't working.abs

    # Trying grid-line heuristic:
    #parser = argparse.ArgumentParser(description="Detect grid-like structure using HoughLines.")
    #parser.add_argument("image", type=str, help="Path to image file")
    #parser.add_argument("--threshold", type=int, default=120, help="Hough line detection threshold")
    #parser.add_argument("--visualize", action="store_true", help="Display image with detected lines")
    #args = parser.parse_args()
    #detect_grid_structure(Path(args.image), hough_thresh=args.threshold, visualize=args.visualize)

    if len(sys.argv) < 2:
        print("Usage: python detect_column_count_standalone.py image.png")
        sys.exit(1)

    image_file = Path(sys.argv[1])
    plot = "--plot" in sys.argv
    detect_column_count(image_file, show_plot=plot)
