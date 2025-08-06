from pathlib import Path
import shutil

PG_DIR = Path("pages")
PNG_DIR = Path("pngs")
REGION_DIR = Path("regions")
SGF_DIR = Path("sgfs")

shutil.rmtree(PG_DIR)
shutil.rmtree(PNG_DIR)
shutil.rmtree(REGION_DIR)
shutil.rmtree(SGF_DIR)

PG_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)
REGION_DIR.mkdir(exist_ok=True)
SGF_DIR.mkdir(exist_ok=True)