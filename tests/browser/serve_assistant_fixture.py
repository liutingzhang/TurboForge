"""Local UI boundary fixture server. Not imported or shipped as a product route."""
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
from flask import send_from_directory
from turboforge.api.server import create_app

app = create_app()


@app.get("/__assistant_fixture/<name>")
def fixture(name):
    if name not in ("index.html", "fixture.js", "fixture.css"):
        return "Not found", 404
    return send_from_directory(ROOT / "reports/assistant-browser", name)


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1",port=8098,debug=False,threaded=True,load_dotenv=False)
