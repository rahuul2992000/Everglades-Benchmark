"""NeuralForecast model service.  Run:  uvicorn service:app --port 8001"""
from everglades_core.service import make_app
from nf_runner import NFRunner

app = make_app(NFRunner())
