__version__ = '0.0.1'

# Cloud module is optional (requires boto3, s3fs, etc.)
try:
    from . import cloud
except ImportError:
    pass