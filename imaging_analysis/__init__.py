import sys
from pathlib import Path
import os

# Add absolute path to monkey_fog root folder
MF_ROOT_PATH = str(Path(__file__).parent.absolute())
sys.path.append(MF_ROOT_PATH)