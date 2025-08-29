# __init__ file
import warnings
warnings.filterwarnings('ignore')

from . import tools as tl
from . import plotting as pl

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl','pl']})