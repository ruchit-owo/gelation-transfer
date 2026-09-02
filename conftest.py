"""Make ``src/`` importable without an editable install.

Lets ``pytest`` and the experiment scripts do ``import gelbench`` whether or not
``pip install -e .`` has been run.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
