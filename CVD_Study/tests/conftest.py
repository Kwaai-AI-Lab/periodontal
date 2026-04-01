import sys
from pathlib import Path

# Add CVD_Study to path so `from generate_cvd_figures import ...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
