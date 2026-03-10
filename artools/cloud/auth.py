"""
Orchestrator authentication helper for NASA Earthdata S3 access.

Call get_gesdisc_s3_credentials() ONCE in the orchestrator before dispatching
Lithops workers. Pass the returned credentials dict to each worker invocation.

Credentials are valid for approximately 1 hour.
"""

import earthaccess


def get_gesdisc_s3_credentials() -> dict:
    """
    Authenticate with NASA Earthdata and return temporary S3 credentials
    for the GES DISC DAAC (where MERRA-2 data is hosted).

    Returns
    -------
    dict
        S3 credentials with keys:
        - accessKeyId: str
        - secretAccessKey: str
        - sessionToken: str
        - expiration: str (ISO timestamp)
    """
    auth = earthaccess.login()
    return auth.get_s3_credentials(daac="GES_DISC")
