__version__ = '0.0.1'

# Cloud module is importable but optional (requires lithops)
try:
    from . import cloud
except ImportError:
    pass