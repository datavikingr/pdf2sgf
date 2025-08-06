from pathlib import Path
import shutil

PG_DIR = Path("pages")
HPG_DIR = Path("page_slices")
PROBLEM_DIR = Path("problems")
SGF_DIR = Path("sgfs")

shutil.rmtree(PG_DIR)
shutil.rmtree(HPG_DIR)
shutil.rmtree(PROBLEM_DIR)
shutil.rmtree(SGF_DIR)

PG_DIR.mkdir(exist_ok=True)
HPG_DIR.mkdir(exist_ok=True)
PROBLEM_DIR.mkdir(exist_ok=True)
SGF_DIR.mkdir(exist_ok=True)