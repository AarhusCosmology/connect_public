import os
import sys
from pathlib import Path

FILE_PATH = os.path.realpath(os.path.dirname(__file__))
CONNECT_PATH = Path(FILE_PATH).parents[1]
print(CONNECT_PATH)

sys.path.insert(1, CONNECT_PATH._str)
print(sys.path)

from source.mcmc_base import MCMC_base_class
