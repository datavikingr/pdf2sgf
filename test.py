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
PNG_DIR = Path("pngs")
REGION_DIR = Path("regions")
SGF_DIR = Path("sgfs")
IMG2SGF_SCRIPT = Path("img2sgf.py")

PG_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)
REGION_DIR.mkdir(exist_ok=True)
SGF_DIR.mkdir(exist_ok=True)

MARGIN = 150 # 0.5in on a 300 dpi page
GAP_THRESHOLD = 25  # rows of white space = separator
MIN_HEIGHT = 100    # ignore tiny noise regions

def is_diagram_page(image, pixel_threshold=100000):
    """
    Returns True if the page has enough black pixels to likely contain Go diagrams.
    `pixel_threshold` may need tuning based on DPI and image size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    black_pixel_count = cv2.countNonZero(thresh)
    return black_pixel_count > pixel_threshold

if __name__ == "__main__":