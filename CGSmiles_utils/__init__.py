'''
Deprecated name of the package, use ``import cgsmiles_utils`` instead.
'''
import warnings

from cgsmiles_utils import *  # noqa: F401,F403
from cgsmiles_utils import __all__, __version__  # noqa: F401

warnings.warn("The package CGSmiles_utils was renamed to cgsmiles_utils, import from cgsmiles_utils instead.",
			  DeprecationWarning, stacklevel=2)
