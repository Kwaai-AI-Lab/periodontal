import sys
from pathlib import Path

# Add AD_Study/v2 to path so `from IBM_PD_AD import ...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v2"))
