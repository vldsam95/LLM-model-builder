from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from src.builder.service import (
    FOUNDATION_PACKS,
    MODEL_PRESETS,
    NEUTRAL_FALLBACK_DIALOGUES,
    PREDICTIVE_TASK_LABELS,
    PROFILE_EXAMPLES,
    PROJECT_FAMILY_LABELS,
    checkpoint_catalog_paths,
    create_or_update_project,
    effective_default_base_checkpoint,
    list_projects,
    load_project_manifest,
    model_catalog,
    project_chat_artifacts,
    predictive_review_path,
    predictive_project_artifacts,
    project_summary,
    read_log_tail,
    refresh_runtime,
    start_training_job,
    write_json,
)
from src.inference.chat_engine import ChatEngine
from src.predictive.tabular import load_predictive_bundle, predict_with_bundle, review_predictive_dataset

builder_bp = Blueprint("builder", __name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


def base_checkpoints() -> list[str]:
    return checkpoint_catalog_paths()


@lru_cache(maxsize=12)
def get_builder_engine(engine_key: str, checkpoint_path: str, tokenizer_path: str, project_fallback_path: str) -> ChatEngine:
    del engine_key
    fallback_paths = [project_fallback_path] if project_fallback_path else []
    fallback_paths.append(str(PROJECT_ROOT / NEUTRAL_FALLBACK_DIALOGUES))
    return ChatEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device="cpu",
        fallback_paths=fallback_paths,
        ruleset="neutral",
    )


def engine_for_project(slug: str) -> ChatEngine:
    checkpoint, tokenizer = project_chat_artifacts(slug)
    project_fallback = PROJECT_ROOT / "builder_projects" / slug / "raw" / "sft_dialogues.jsonl"
    fallback_key = f"{project_fallback}:{project_fallback.stat().st_mtime_ns}" if project_fallback.exists() else "no-project-fallback"
    key = f"{checkpoint}:{checkpoint.stat().st_mtime_ns}:{tokenizer}:{tokenizer.stat().st_mtime_ns}:{fallback_key}"
    return get_builder_engine(key, str(checkpoint), str(tokenizer), str(project_fallback))


@lru_cache(maxsize=12)
def get_predictive_bundle(bundle_key: str, model_path: str) -> dict:
    del bundle_key
    return load_predictive_bundle(model_path)


def predictive_bundle_for_project(slug: str) -> tuple[dict, dict]:
    model_path, evaluation = predictive_project_artifacts(slug)
    key = f"{model_path}:{model_path.stat().st_mtime_ns}"
    return get_predictive_bundle(key, str(model_path)), evaluation


@builder_bp.route("/builder")
def builder_index():
    return render_template(
        "builder.html",
        base_checkpoints=base_checkpoints(),
        projects=list_projects(),
        default_base_checkpoint=effective_default_base_checkpoint(),
        foundation_packs=FOUNDATION_PACKS,
        model_presets=[{"value": key, **value} for key, value in MODEL_PRESETS.items()],
        profile_examples=PROFILE_EXAMPLES,
        project_families=[{"value": key, "label": value} for key, value in PROJECT_FAMILY_LABELS.items()],
        predictive_tasks=[{"value": key, "label": value} for key, value in PREDICTIVE_TASK_LABELS.items()],
        model_catalog=model_catalog(),
    )


@builder_bp.route("/builder/projects/<slug>/chat")
def builder_chat(slug: str):
    project = project_summary(slug)
    if project.get("project_family") == "predictive_ai":
        return redirect(url_for("builder.builder_predictive", slug=slug))
    return render_template("builder_chat.html", project=project)


@builder_bp.route("/builder/projects/<slug>/predictive")
def builder_predictive(slug: str):
    project = project_summary(slug)
    if project.get("project_family") != "predictive_ai":
        return redirect(url_for("builder.builder_chat", slug=slug))

    refresh_runtime(slug)
    _, evaluation = predictive_bundle_for_project(slug) if project.get("active_model") else ({}, {})
    return render_template("builder_predictive.html", project=project, evaluation=evaluation)


@builder_bp.route("/api/builder/projects")
def api_builder_projects():
    return jsonify({"projects": list_projects()})


@builder_bp.route("/api/builder/models")
def api_builder_models():
    return jsonify({"models": model_catalog()})


@builder_bp.route("/api/builder/projects", methods=["POST"])
def api_builder_create_project():
    payload = {key: value for key, value in request.form.items()}
    files = request.files.getlist("files") + request.files.getlist("supporting_files")
    dataset_file = request.files.get("dataset_file")
    project = create_or_update_project(payload, files, dataset_file=dataset_file)
    return jsonify({"project": project})


@builder_bp.route("/api/builder/projects/<slug>")
def api_builder_project(slug: str):
    refresh_runtime(slug)
    return jsonify({"project": project_summary(slug)})


@builder_bp.route("/api/builder/projects/<slug>/train", methods=["POST"])
def api_builder_train(slug: str):
    start_training_job(slug)
    return jsonify({"project": project_summary(slug)})


@builder_bp.route("/api/builder/projects/<slug>/status")
def api_builder_status(slug: str):
    refresh_runtime(slug)
    return jsonify({"project": project_summary(slug)})


@builder_bp.route("/api/builder/projects/<slug>/review")
def api_builder_review(slug: str):
    manifest = load_project_manifest(slug)
    if manifest.get("project_family") != "predictive_ai":
        return jsonify({"review": None, "error": "Dataset review is available only for predictive AI projects."}), 400
    dataset_rel = manifest.get("dataset_path")
    if not dataset_rel:
        return jsonify({"review": None, "error": "No dataset is attached to this project yet."}), 400

    review = review_predictive_dataset(manifest, PROJECT_ROOT / dataset_rel)
    write_json(predictive_review_path(slug), review)
    return jsonify({"review": review})


@builder_bp.route("/api/builder/projects/<slug>/logs")
def api_builder_logs(slug: str):
    refresh_runtime(slug)
    return jsonify({"log_tail": read_log_tail(slug)})


@builder_bp.route("/api/builder/projects/<slug>/chat", methods=["POST"])
def api_builder_chat(slug: str):
    payload = request.get_json(force=True)
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message is empty."}), 400

    manifest = load_project_manifest(slug)
    if manifest.get("project_family") == "predictive_ai":
        return jsonify({"error": "This project is predictive, not chat-based. Open the predictive dashboard instead."}), 400
    engine = engine_for_project(slug)
    response = engine.chat(
        user_text=message,
        system_text=str(payload.get("system_prompt", "")).strip() or manifest.get("system_prompt") or None,
        temperature=float(payload.get("temperature", 0.8)),
        top_k=int(payload.get("top_k", 20)),
        max_new_tokens=int(payload.get("max_new_tokens", 64)),
        json_mode=bool(payload.get("json_mode", False)),
    )
    return jsonify({"response": response})


@builder_bp.route("/api/builder/projects/<slug>/predict", methods=["POST"])
def api_builder_predict(slug: str):
    manifest = load_project_manifest(slug)
    if manifest.get("project_family") != "predictive_ai":
        return jsonify({"error": "Prediction API is available only for predictive AI projects."}), 400

    payload = request.get_json(force=True) or {}
    features = payload.get("features", {})
    if not isinstance(features, dict):
        return jsonify({"error": "features must be a JSON object."}), 400

    bundle, _ = predictive_bundle_for_project(slug)
    response = predict_with_bundle(bundle, features)
    return jsonify({"response": response})
