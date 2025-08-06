from pathlib import Path
import shutil

PG_DIR = Path("pages")
PNG_DIR = Path("pngs")
PROBLEM_DIR = Path("problems")
SGF_DIR = Path("sgfs")

shutil.rmtree(PG_DIR)
shutil.rmtree(PNG_DIR)
shutil.rmtree(PROBLEM_DIR)
shutil.rmtree(SGF_DIR)

PG_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)
PROBLEM_DIR.mkdir(exist_ok=True)
SGF_DIR.mkdir(exist_ok=True)