# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Entry point to Flask-Connexion API
"""

import connexion
from connexion.middleware import MiddlewarePosition
from starlette.middleware.cors import CORSMiddleware

from .astrometry_readiness.start_astrometry_background_check import (
    start_astrometry_background_check,
)

# from . import __version__ as version
# from .config.logging import get_logger
# from .config.env import ENV
# from .config.exceptions import SBNSISException
# from .services.database_provider import db_session

# logger: logging.Logger = get_logger()
app = connexion.FlaskApp(__name__, specification_dir="api/")

app.add_middleware(
    CORSMiddleware,
    position=MiddlewarePosition.BEFORE_EXCEPTION,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_api(
    "openapi.yaml",
    arguments={
        # "version": str(version),
        # "base_href": ENV.BASE_HREF,
    },
)
application = app.app


if __name__ == "__main__":
    start_astrometry_background_check()
    app.run(host="0.0.0.0", port=8000)
