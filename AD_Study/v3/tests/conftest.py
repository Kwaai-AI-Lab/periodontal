import sys
from pathlib import Path

# Add AD_Study/v3 to path so `from IBM_PD_AD_v3 import ...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
