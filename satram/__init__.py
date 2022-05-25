"""A python implementation of MBAR and TRAM and their respective stochastic aproximators SAMBAR and SATRAM"""

from .estimators import ThermodynamicEstimator

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
