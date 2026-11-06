from __future__ import annotations

import json
import traceback
from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builder.service import (
    load_project_manifest,
    project_dir,
    runtime_path,
    utc_now,
    update_runtime_failure,
    write_json,
)
from src.predictive.tabular import save_predictive_bundle, train_predictive_bundle

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run a predictive builder project training job.")
    parser.add_argument("--project-slug", required=True)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    slug = args.project_slug
    manifest = load_project_manifest(slug)
    dataset_rel = manifest.get("dataset_path")
    if not dataset_rel:
        raise ValueError("Predictive project is missing dataset_path in manifest.")

    dataset_path = PROJECT_ROOT / dataset_rel
    bundle = train_predictive_bundle(manifest, dataset_path)

    project_root = project_dir(slug)
    artifact_dir = project_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "predictive_model.joblib"
    evaluation_path = artifact_dir / "evaluation.json"

    save_predictive_bundle(bundle, model_path)
    evaluation_path.write_text(
        json.dumps(bundle["evaluation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runtime = {
        "status": "completed",
        "pid": None,
        "started_at": None,
        "completed_at": utc_now(),
        "error": None,
        "log_path": str(project_root / "logs" / "train.log"),
        "active_model": str(model_path.relative_to(PROJECT_ROOT)),
        "evaluation_path": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "active_checkpoint": None,
        "active_tokenizer": None,
    }
    write_json(runtime_path(slug), runtime)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        slug = None
        try:
            parser = parse_args()
            slug = parser.parse_args().project_slug
        except Exception:
            slug = None
        if slug:
            update_runtime_failure(slug, traceback.format_exc())
        raise
