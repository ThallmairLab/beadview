'''
Utilities for CGSmiles mappings: drawing, structure generation and ITP editing.
'''
from .beads import BeadType
from .drawing.layout import parallel_tails
from .drawing.options import DrawOptions
from .mapping import CGMapping
from .topology import ITPConsistencyWarning, Topology

__version__ = '0.1.0'

__all__ = ['CGMapping', 'DrawOptions', 'parallel_tails', 'BeadType', 'Topology', 'ITPConsistencyWarning', '__version__']
