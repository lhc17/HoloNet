"""

Python library to decode functional cell–cell communication events
by multi-view graph learning on spatial transcriptomics

"""

from . import plotting as pl
from . import predicting as pr
from . import preprocessing as pp
from . import tools as tl
from ._setting import *
from .colorSchemes import *

from ._metadata import __version__, __author__, __email__, within_flit
