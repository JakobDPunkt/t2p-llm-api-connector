from flask import Blueprint

bp = Blueprint("api", __name__)

from app.api import routes  # noqa: F401,E402
from app.api import pnml_routes  # noqa: F401,E402
