import sys
from pathlib import Path

__project_root__ = Path(__file__).resolve().parents[1]
sys.path.append(str(__project_root__))

print(f"Project root is {__project_root__}")
