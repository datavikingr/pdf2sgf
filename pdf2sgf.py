import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import subprocess
import os
import shutil

PDF_DIR = Path("pdfs")
PG_DIR = Path("pages")
HPG_DIR = Path("page_slices")
PROBLEM_DIR = Path("problems")
SGF_DIR = Path("sgfs")
IMG2SGF_SCRIPT = Path("img2sgf.py")

PG_DIR.mkdir(exist_ok=True)
HPG_DIR.mkdir(exist_ok=True)
PROBLEM_DIR.mkdir(exist_ok=True)
SGF_DIR.mkdir(exist_ok=True)

MARGIN = 150 # 0.5in on a 300 dpi page
GAP_THRESHOLD = 25  # rows of white space = separator
MIN_HEIGHT = 100    # ignore tiny noise regions

def crop_margins(img: Image.Image) -> Image.Image:
    width, height = img.size
    cropped = img.crop((
        MARGIN,              # left
        MARGIN,              # top
        width - MARGIN,      # right
        height - MARGIN      # bottom
    ))
    return cropped

def split_vertically(img: Image.Image):
    width, height = img.size
    mid = width // 2
    left_half = img.crop((0, 0, mid, height))
    right_half = img.crop((mid, 0, width, height))
    return left_half, right_half

def process_pdf(pdf_path):
    print(f"\nProcessing {pdf_path.name}")
    images = convert_from_path(pdf_path, dpi=300)
    for page_num, image in enumerate(images):
        base_name = f"{pdf_path.stem}_page{page_num+1}"
        full_img_path = PG_DIR / f"{base_name}.png"
        image.save(full_img_path)
        cropped = crop_margins(image) # Step 1: Crop margins
        left, right = split_vertically(cropped) # Step 2: Split into left/right
        # Step 3: Save halves
        left_path = HPG_DIR / f"{base_name}_left.png"
        right_path = HPG_DIR / f"{base_name}_right.png"
        left.save(left_path)
        right.save(right_path)
        print(f"Saved: {left_path.name}, {right_path.name}")

def detect_problem_regions(image_path: Path, counter: list):
    print(f"\nDetecting in {image_path.name}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Binarize: white = 255, black = 0
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Row-wise sum of black pixels
    row_sums = np.sum(thresh == 0, axis=1)  # black pixels per row

    # Find active regions (rows that contain ink)
    regions = []
    in_region = False
    start = 0

    for i, val in enumerate(row_sums):
        if val > 0 and not in_region:
            in_region = True
            start = i
        elif val == 0 and in_region:
            if i - start >= MIN_HEIGHT:
                regions.append((start, i))
            in_region = False

    # Merge regions separated by less than GAP_THRESHOLD
    merged = []
    if not regions:
        print(f"No active regions found in {image_path.name}")
        return
    prev_start, prev_end = regions[0]
    for start, end in regions[1:]:
        if start - prev_end < GAP_THRESHOLD:
            prev_end = end
        else:
            merged.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    merged.append((prev_start, prev_end))

    # Save cropped images
    img_rgb = cv2.imread(str(image_path))  # original for saving
    for idx, (y1, y2) in enumerate(merged):
        crop = img_rgb[y1:y2, :]
        #out_path = PROBLEM_DIR / f"{image_path.stem}_prob{idx+1}.png"
        out_path = PROBLEM_DIR / f"problem{counter[0]}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved: {out_path.name}")
        counter[0] += 1

if __name__ == "__main__":
    for pdf_file in PDF_DIR.glob("*.pdf"): # Process all PDFs
        process_pdf(pdf_file)
    problem_counter = [1]
    for img_file in sorted(HPG_DIR.glob("*.png")):
        detect_problem_regions(img_file, problem_counter)
    for prob_file in sorted(PROBLEM_DIR.glob("*.png")):
        output_path = SGF_DIR / prob_file.with_suffix(".sgf").name
        player = "black"
        print(f"Converting {prob_file.name} → {output_path.name}")
        try:
            subprocess.run(["python", "img2sgf.py", str(prob_file), str(output_path), player])
            print(f"Saved: {output_path.name}")
        except Exception as e:
            print(f"Exception while processing {prob_file.name}: {e}")
            continue
    