# -*- coding: utf-8 -*-
"""nb_tools.py — build and execute the notebooks from lists of cells.

Each notebook is described as a list of tuples ("md" | "code", text). It is
then executed end to end with nbclient: outputs and figures are embedded in
the delivered .ipynb file.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

CODE_HEADER = """# --- Environment shared by all notebooks ------------------------------------
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MODULE as m                # core module of the study (fully commented)
import figure_style as fs         # visual style of the paper's figures
fs.apply()
pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 4)
"""


def code_header(module: str) -> str:
    return CODE_HEADER.replace("MODULE", module)


def build(name: str, cells: list[tuple[str, str]], execute: bool = True, timeout: int = 7200) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, text in cells:
        text = text.strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(text) if kind == "md"
                        else nbformat.v4.new_code_cell(text))
    path = ROOT / "notebooks" / f"{name}.ipynb"
    path.parent.mkdir(exist_ok=True)
    if execute:
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    return path
