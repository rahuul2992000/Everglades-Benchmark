"""Chronos-Bolt model service.  Run:  uvicorn service:app --port 8002"""
from everglades_core.service import make_app
from chronos_runner import ChronosRunner

app = make_app(ChronosRunner())
