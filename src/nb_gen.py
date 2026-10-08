import nbformat as nbf

nb = nbf.v4.new_notebook()
text = """# Deep Learning Application in Data Science
This notebook reproduces the main results of the project using the source code from `src/`."""

code = """import sys
import os
sys.path.append(os.path.abspath('../src'))

import run
run.mn()"""

nb['cells'] = [nbf.v4.new_markdown_cell(text), nbf.v4.new_code_cell(code)]

with open("notebooks/deep_learning.ipynb", "w") as f:
    nbf.write(nb, f)
