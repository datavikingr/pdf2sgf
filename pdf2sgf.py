import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import subprocess
import os
import shutil
import pytesseract

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

def split_vertically(img: Image.Image): #  -> Image.Image, Image.Image
    width, height = img.size
    mid = width // 2
    left_half = img.crop((0, 0, mid, height))
    right_half = img.crop((mid, 0, width, height))
    return left_half, right_half

def split_thirds_vertically(img: Image.Image): #  -> Image.Image, Image.Image
    width, height = img.size
    left_mid = width // 3
    right_mid = left_mid * 2
    left_half = img.crop((0, 0, left_mid, height))
    center_half = img.crop((left_mid, 0, right_mid, height))
    right_half = img.crop((right_mid, 0, width, height))
    return left_half, center_half, right_half

def detect_player_from_text(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img).lower()
    if "white to play" in text or "white" in text:
        return "white"
    elif "black to play" in text or "black" in text:
        return "black"
    else:
        return "black"  # default fallback

def detect_column_count(image) -> int:
    #Accepts a PIL.Image.Image or np.ndarray.
    # Convert to grayscale NumPy array if needed
    if hasattr(image, 'convert'):
        image = np.array(image.convert("L"))  # Convert PIL to grayscale NumPy
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return col_count

def process_pdf(pdf_path):
    print(f"\nProcessing {pdf_path.name}")
    images = convert_from_path(pdf_path, dpi=300) #Get the pdf
    for page_num, image in enumerate(images): #for each page
        base_name = f"{pdf_path.stem}_page{page_num+1}" #build new file name for individual pages
        full_img_path = PG_DIR / f"{base_name}.png" #build full file path
        image.save(full_img_path) #save each page
        # REAL PROCESSING TIME
        cropped = crop_margins(image) # Step 1: Crop margins out
        columns = detect_column_count(cropped) # Step 2: detect columns for logic
        if columns == 2: # if two, we slice in half vertically
            left, right = split_vertically(cropped) # Step 3: Split into left/right
            left_path = HPG_DIR / f"{base_name}_left.png" #build left file paths
            right_path = HPG_DIR / f"{base_name}_right.png" #build right file paths
            left.save(left_path) #save left
            right.save(right_path) #save right - this is feeling like a Twix commercial
            print(f"Saved: {left_path.name}, {right_path.name}")
        elif columns ==3: # if 3, we slice into vertical thirds
            left, right, center = split_thirds_vertically(cropped) # Step 3: Split into left/right/center
            left_path = HPG_DIR / f"{base_name}_left.png" #build left file paths
            right_path = HPG_DIR / f"{base_name}_right.png" #build right file paths
            center_path = HPG_DIR / f"{base_name}_center.png" #build center file paths
            left.save(left_path) #save left
            right.save(right_path) #save right
            center.save(left_path) #save center
            print(f"Saved: {left_path.name}, {right_path.name}, {center_path.name}")
        else:
            print(f"Something deeply fucky is happening here, skipping {pdf_path.name}.")
            continue

def detect_problem_regions(image_path: Path, counter: list):
    print(f"\nDetecting in {image_path.name}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) # Binarize: white = 255, black = 0
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
        ### 🧠 NEW GRID HEURISTIC
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_crop, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        num_lines = 0 if lines is None else len(lines)
        if num_lines < 15:
            print(f"⚠️ Skipping region {idx+1} — too few grid lines ({num_lines})")
            continue
        #out_path = PROBLEM_DIR / f"{image_path.stem}_prob{idx+1}.png"
        out_path = PROBLEM_DIR / f"problem{counter[0]}.png"
        cv2.imwrite(str(out_path), crop)
        # Crop footer band for OCR
        footer_height = 50
        footer_band = img_rgb[y2:y2 + footer_height, :]
        footer_img = Image.fromarray(footer_band)
        player = detect_player_from_text(footer_img)
        # Save player info alongside image, e.g. as a sidecar txt file:
        player_path = PROBLEM_DIR / f"{out_path.stem}.player"
        player_path.write_text(player)
        print(f"Saved: {out_path.name} ({player})")
        counter[0] += 1

if __name__ == "__main__":
    for pdf_file in PDF_DIR.glob("*.pdf"): # Process all PDFs
        process_pdf(pdf_file)
    problem_counter = [1]
    for img_file in sorted(HPG_DIR.glob("*.png")):
        detect_problem_regions(img_file, problem_counter)
    for prob_file in sorted(PROBLEM_DIR.glob("*.png")):
        output_path = SGF_DIR / prob_file.with_suffix(".sgf").name
        player_path = prob_file.with_suffix(".player")
        if player_path.exists():
            player = player_path.read_text().strip()
        else:
            player = "black"
        print(f"Converting {prob_file.name} → {output_path.name}")
        try:
            subprocess.run(["python", "img2sgf.py", str(prob_file), str(output_path), player])
            print(f"Saved: {output_path.name}")
        except Exception as e:
            print(f"Exception while processing {prob_file.name}: {e}")
            continue
    