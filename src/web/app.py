from __future__ import annotations

import argparse

from flask import Flask

from src.web.builder_routes import builder_bp
from src.web.routes import web_bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/builder/static",
    )
    app.register_blueprint(web_bp)
    app.register_blueprint(builder_bp)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
