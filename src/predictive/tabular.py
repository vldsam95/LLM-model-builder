from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SUPPORTED_DATASET_SUFFIXES = {".csv", ".xlsx", ".xlsm"}
PREDICTIVE_TASKS = {"classification", "regression", "forecasting"}


@dataclass
class DatasetOverview:
    rows: int
    columns: int
    column_names: list[str]
    preview: list[dict[str, Any]]
    dtypes: dict[str, str]


def load_tabular_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(dataset_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.reset_index(drop=True)


def inspect_tabular_dataset(path: str | Path) -> DatasetOverview:
    frame = load_tabular_dataset(path)
    preview = frame.head(5).replace({np.nan: None}).to_dict(orient="records")
    dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
    return DatasetOverview(
        rows=int(len(frame)),
        columns=int(len(frame.columns)),
        column_names=[str(column) for column in frame.columns],
        preview=preview,
        dtypes=dtypes,
    )


def review_predictive_dataset(manifest: dict[str, Any], dataset_path: str | Path) -> dict[str, Any]:
    frame = load_tabular_dataset(dataset_path)
    task = str(manifest.get("predictive_task", "regression")).strip() or "regression"
    target_column = str(manifest.get("target_column", "")).strip()
    time_column = str(manifest.get("time_column", "")).strip() or None
    requested_features = _parse_features(manifest.get("feature_columns"))
    feature_columns = requested_features or [column for column in frame.columns if column != target_column]
    feature_columns = [column for column in feature_columns if column in frame.columns and column != target_column]

    missing = []
    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        if missing_count:
            missing.append(
                {
                    "column": str(column),
                    "missing": missing_count,
                    "ratio": round(float(missing_count / max(len(frame), 1)), 4),
                }
            )
    missing.sort(key=lambda item: item["missing"], reverse=True)

    duplicate_rows = int(frame.duplicated().sum())
    constant_columns = [str(column) for column in frame.columns if frame[column].nunique(dropna=False) <= 1]
    object_columns = [str(column) for column in frame.columns if frame[column].dtype == "object"]
    high_cardinality = []
    threshold = max(20, min(80, int(len(frame) * 0.2)))
    for column in object_columns:
        unique_count = int(frame[column].nunique(dropna=True))
        if unique_count >= threshold:
            high_cardinality.append({"column": column, "unique_values": unique_count})

    warnings: list[str] = []
    recommendations: list[str] = []

    if len(frame) < 120:
        warnings.append("The dataset is quite small. Results may vary strongly across splits.")
        recommendations.append("Collect more rows before trusting the model in a business workflow.")
    elif len(frame) < 300:
        recommendations.append("The dataset is usable, but more rows would likely improve generalization.")

    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate rows were detected.")
        recommendations.append("Deduplicate the dataset before the final training run.")

    if missing:
        warnings.append("Some columns contain missing values.")
        recommendations.append("Review columns with many missing values and decide whether to impute, clean, or remove them.")

    if constant_columns:
        warnings.append(f"Constant columns detected: {', '.join(constant_columns[:4])}.")
        recommendations.append("Remove constant columns because they do not add predictive signal.")

    if high_cardinality:
        warnings.append("High-cardinality categorical columns were detected.")
        recommendations.append("Consider grouping rare categories or using cleaner code tables for very wide categorical columns.")

    target_summary: dict[str, Any] = {}
    if target_column and target_column in frame.columns:
        target_series = frame[target_column].dropna()
        if task == "classification":
            counts = target_series.astype(str).value_counts().to_dict()
            class_counts = {str(key): int(value) for key, value in counts.items()}
            ratios = list(class_counts.values())
            imbalance_ratio = round(min(ratios) / max(ratios), 4) if len(ratios) > 1 and max(ratios) else 1.0
            target_summary = {"class_counts": class_counts, "imbalance_ratio": imbalance_ratio}
            if len(class_counts) < 2:
                warnings.append("The classification target has fewer than two classes.")
            elif imbalance_ratio < 0.2:
                warnings.append("The target classes are highly imbalanced.")
                recommendations.append("Consider collecting more samples for the rare class or evaluating with balanced metrics.")
        else:
            numeric_target = pd.to_numeric(target_series, errors="coerce").dropna()
            if not numeric_target.empty:
                target_summary = {
                    "min": round(float(numeric_target.min()), 4),
                    "max": round(float(numeric_target.max()), 4),
                    "mean": round(float(numeric_target.mean()), 4),
                    "std": round(float(numeric_target.std(ddof=0)), 4),
                }
            if task == "forecasting" and not time_column:
                warnings.append("Forecasting was selected without a time column.")

    time_summary: dict[str, Any] = {}
    if time_column:
        if time_column not in frame.columns:
            warnings.append(f"Time column '{time_column}' does not exist in the dataset.")
            recommendations.append("Set a valid time column before training a forecasting project.")
        else:
            parsed = pd.to_datetime(frame[time_column], errors="coerce")
            invalid_time = int(parsed.isna().sum())
            time_duplicates = int(parsed.dropna().duplicated().sum())
            inferred_freq = pd.infer_freq(parsed.dropna().sort_values()) if len(parsed.dropna()) >= 3 else None
            time_summary = {
                "invalid_timestamps": invalid_time,
                "duplicate_timestamps": time_duplicates,
                "inferred_frequency": inferred_freq,
            }
            if invalid_time:
                warnings.append("Some timestamps could not be parsed.")
                recommendations.append("Normalize the time column to a consistent datetime format.")
            if task == "forecasting" and time_duplicates:
                recommendations.append("For forecasting, duplicate timestamps should usually be aggregated or disambiguated.")

    if not requested_features:
        recommendations.append("Feature columns were left empty, so the builder will use all columns except the target automatically.")

    readiness_score = 100
    readiness_score -= 18 if len(frame) < 120 else 8 if len(frame) < 300 else 0
    readiness_score -= min(20, len(missing) * 4)
    readiness_score -= 10 if duplicate_rows else 0
    readiness_score -= 8 if high_cardinality else 0
    readiness_score = max(35, readiness_score)

    if readiness_score >= 85:
        readiness_label = "Strong"
    elif readiness_score >= 65:
        readiness_label = "Usable"
    else:
        readiness_label = "Needs cleanup"

    return {
        "task": task,
        "target_column": target_column,
        "time_column": time_column,
        "feature_columns": feature_columns,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "high_cardinality": high_cardinality,
        "missing": missing[:10],
        "warnings": warnings,
        "recommendations": recommendations,
        "target_summary": target_summary,
        "time_summary": time_summary,
        "preview": _clean_records(frame, limit=5),
        "readiness_score": readiness_score,
        "readiness_label": readiness_label,
    }


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").replace("\u200f", " ").replace("\u200e", " ").split()).strip()


def _parse_features(raw: list[str] | str | None) -> list[str]:
    if isinstance(raw, list):
        return [_normalize_space(item) for item in raw if _normalize_space(item)]
    if raw is None:
        return []
    return [_normalize_space(part) for part in str(raw).replace("\n", ",").split(",") if _normalize_space(part)]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _clean_records(frame: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.head(limit).replace({np.nan: None}).to_dict(orient="records")
    ]


def _train_val_split(
    frame: pd.DataFrame,
    *,
    target_column: str,
    task: str,
    time_column: str | None = None,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if time_column and time_column in frame.columns:
        ordered = frame.copy()
        ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce")
        ordered = ordered.dropna(subset=[time_column])
        ordered = ordered.sort_values(time_column).reset_index(drop=True)
        split_index = max(1, int(len(ordered) * (1 - val_ratio)))
        return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()

    stratify = None
    if task == "classification":
        target_counts = frame[target_column].value_counts()
        if len(target_counts) > 1 and int(target_counts.min()) >= 2:
            stratify = frame[target_column]

    train_frame, val_frame = train_test_split(
        frame,
        test_size=val_ratio,
        random_state=42,
        stratify=stratify,
    )
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)


def _expand_time_features(frame: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    if not time_column or time_column not in frame.columns:
        return frame

    enriched = frame.copy()
    parsed = pd.to_datetime(enriched[time_column], errors="coerce")
    enriched[f"{time_column}__year"] = parsed.dt.year
    enriched[f"{time_column}__month"] = parsed.dt.month
    enriched[f"{time_column}__day"] = parsed.dt.day
    enriched[f"{time_column}__weekday"] = parsed.dt.weekday
    enriched[f"{time_column}__hour"] = parsed.dt.hour
    enriched = enriched.drop(columns=[time_column])
    return enriched


def _build_supervised_table(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    time_column: str | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    prepared = frame.copy()
    prepared = prepared.dropna(subset=[target_column]).reset_index(drop=True)
    features = feature_columns or [column for column in prepared.columns if column != target_column]
    features = [column for column in features if column in prepared.columns and column != target_column]
    if time_column and time_column in prepared.columns and time_column not in features:
        features.append(time_column)
    model_frame = prepared[features + [target_column]].copy()
    return model_frame[features], model_frame[target_column], features


def _build_forecasting_table(
    frame: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
    feature_columns: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[dict[str, Any]]]:
    if time_column not in frame.columns:
        raise ValueError("Forecasting requires a valid time column.")

    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce")
    ordered = ordered.dropna(subset=[time_column, target_column]).sort_values(time_column).reset_index(drop=True)
    if len(ordered) < 24:
        raise ValueError("Forecasting requires at least 24 rows after cleaning.")

    candidate_lags = [1, 2, 3, 6, 12, 24]
    lag_steps = [lag for lag in candidate_lags if lag < len(ordered) - horizon]
    if not lag_steps:
        lag_steps = [1, 2, 3]

    supervised = ordered.copy()
    extra_features = [column for column in feature_columns if column in supervised.columns and column not in {target_column, time_column}]

    for lag in lag_steps:
        supervised[f"{target_column}__lag_{lag}"] = supervised[target_column].shift(lag)

    for window in [3, 6]:
        if window < len(ordered) - horizon:
            supervised[f"{target_column}__roll_mean_{window}"] = supervised[target_column].shift(1).rolling(window=window).mean()

    supervised[f"{time_column}__year"] = supervised[time_column].dt.year
    supervised[f"{time_column}__month"] = supervised[time_column].dt.month
    supervised[f"{time_column}__day"] = supervised[time_column].dt.day
    supervised[f"{time_column}__weekday"] = supervised[time_column].dt.weekday
    supervised[f"{time_column}__hour"] = supervised[time_column].dt.hour
    supervised["target_future"] = supervised[target_column].shift(-horizon)

    final_features = [
        column
        for column in supervised.columns
        if column not in {target_column, "target_future", time_column}
        and (column in extra_features or column.startswith(f"{target_column}__") or column.startswith(f"{time_column}__"))
    ]
    model_frame = supervised[final_features + ["target_future", time_column]].dropna().reset_index(drop=True)
    if len(model_frame) < 16:
        raise ValueError("Forecasting left too few usable rows after lag generation.")

    history_tail = _clean_records(ordered.tail(max(lag_steps) + 5))
    return (
        model_frame[final_features],
        model_frame["target_future"],
        model_frame[[time_column]].copy(),
        final_features,
        history_tail,
    )


def _feature_schema(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in feature_columns:
        series = frame[column] if column in frame.columns else pd.Series(dtype="object")
        if pd.api.types.is_numeric_dtype(series):
            kind = "number"
            example = _safe_float(series.dropna().iloc[0]) if not series.dropna().empty else None
        else:
            kind = "text"
            example = str(series.dropna().iloc[0]) if not series.dropna().empty else None
        schema.append({"name": column, "kind": kind, "example": example})
    return schema


def _build_pipeline(feature_frame: pd.DataFrame, task: str) -> Pipeline:
    numeric_columns = [column for column in feature_frame.columns if pd.api.types.is_numeric_dtype(feature_frame[column])]
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    if task == "classification":
        estimator = RandomForestClassifier(
            n_estimators=260,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=320,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray | None, labels: list[str]) -> dict[str, Any]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }
    confusion = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"metrics": metrics, "labels": labels, "confusion_matrix": confusion, "probabilities": probabilities}


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    rmse = math.sqrt(float(mean_squared_error(y_true, y_pred)))
    return {
        "metrics": {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(rmse, 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
    }


def _forecast_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    rmse = math.sqrt(float(mean_squared_error(y_true, y_pred)))
    safe_true = np.where(np.asarray(y_true) == 0, 1e-6, np.asarray(y_true))
    mape = mean_absolute_percentage_error(safe_true, y_pred)
    return {
        "metrics": {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(rmse, 4),
            "mape": round(float(mape), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
    }


def _compute_feature_importance(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    task: str,
) -> list[dict[str, Any]]:
    if features.empty:
        return []

    sample_size = min(len(features), 220)
    sample_features = features.head(sample_size)
    sample_target = target.head(sample_size)
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"

    importance = permutation_importance(
        pipeline,
        sample_features,
        sample_target,
        n_repeats=4,
        random_state=42,
        scoring=scoring,
        n_jobs=1,
    )
    rows = []
    for name, value in zip(sample_features.columns, importance.importances_mean, strict=False):
        rows.append({"feature": str(name), "importance": round(float(value), 6)})
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows[:10]


def _predictive_summary(task: str, metrics: dict[str, float], top_features: list[dict[str, Any]], target_column: str, horizon: int) -> str:
    feature_text = ", ".join(item["feature"] for item in top_features[:3]) or "no dominant features yet"
    if task == "classification":
        return (
            f"המודל הוא מסווג טבלאי. Accuracy={metrics.get('accuracy', 0):.2f}, "
            f"F1={metrics.get('f1_macro', 0):.2f}. הפיצ'רים המשפיעים ביותר הם {feature_text}."
        )
    if task == "forecasting":
        return (
            f"המודל חוזה את {target_column} {horizon} צעדים קדימה. "
            f"RMSE={metrics.get('rmse', 0):.2f}, MAE={metrics.get('mae', 0):.2f}. "
            f"הפיצ'רים הבולטים הם {feature_text}."
        )
    return (
        f"המודל הוא רגרסור טבלאי. RMSE={metrics.get('rmse', 0):.2f}, "
        f"MAE={metrics.get('mae', 0):.2f}. הפיצ'רים הבולטים הם {feature_text}."
    )


def _future_forecast(
    pipeline: Pipeline,
    source_frame: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
    feature_columns: list[str],
    horizon: int,
) -> list[dict[str, Any]]:
    ordered = source_frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce")
    ordered = ordered.dropna(subset=[time_column, target_column]).sort_values(time_column).reset_index(drop=True)
    if ordered.empty:
        return []

    freq = pd.infer_freq(ordered[time_column])
    if not freq and len(ordered) >= 2:
        delta = ordered[time_column].iloc[-1] - ordered[time_column].iloc[-2]
        freq = delta if delta != pd.Timedelta(0) else pd.Timedelta(hours=1)
    if not freq:
        freq = pd.Timedelta(hours=1)

    candidate_lags = [1, 2, 3, 6, 12, 24]
    lag_steps = [lag for lag in candidate_lags if lag < len(ordered)]
    lag_steps = lag_steps or [1, 2, 3]
    history_target = list(ordered[target_column].astype(float))
    future_rows: list[dict[str, Any]] = []
    last_row = ordered.iloc[-1].to_dict()

    for step in range(1, horizon + 1):
        timestamp = ordered[time_column].iloc[-1] + step * (pd.tseries.frequencies.to_offset(freq) if isinstance(freq, str) else freq)
        row: dict[str, Any] = {}
        for feature in feature_columns:
            if feature.startswith(f"{target_column}__lag_"):
                lag = int(feature.rsplit("_", 1)[-1])
                row[feature] = history_target[-lag] if lag <= len(history_target) else history_target[0]
            elif feature.startswith(f"{target_column}__roll_mean_"):
                window = int(feature.rsplit("_", 1)[-1])
                row[feature] = float(np.mean(history_target[-window:])) if len(history_target) >= window else float(np.mean(history_target))
            elif feature == f"{time_column}__year":
                row[feature] = timestamp.year
            elif feature == f"{time_column}__month":
                row[feature] = timestamp.month
            elif feature == f"{time_column}__day":
                row[feature] = timestamp.day
            elif feature == f"{time_column}__weekday":
                row[feature] = timestamp.weekday()
            elif feature == f"{time_column}__hour":
                row[feature] = timestamp.hour
            else:
                row[feature] = last_row.get(feature)

        prediction = float(pipeline.predict(pd.DataFrame([row]))[0])
        history_target.append(prediction)
        future_rows.append({"time": timestamp.isoformat(), "predicted": round(prediction, 4)})

    return future_rows


def train_predictive_bundle(manifest: dict[str, Any], dataset_path: str | Path) -> dict[str, Any]:
    task = str(manifest.get("predictive_task", "regression")).strip() or "regression"
    if task not in PREDICTIVE_TASKS:
        raise ValueError(f"Unsupported predictive task: {task}")

    target_column = str(manifest.get("target_column", "")).strip()
    if not target_column:
        raise ValueError("Target column is required for predictive AI projects.")

    time_column = str(manifest.get("time_column", "")).strip() or None
    horizon = max(1, int(manifest.get("forecast_horizon", 3) or 3))
    source_frame = load_tabular_dataset(dataset_path)
    if target_column not in source_frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    if time_column and time_column not in source_frame.columns:
        raise ValueError(f"Time column '{time_column}' was not found in the dataset.")

    requested_features = _parse_features(manifest.get("feature_columns"))

    if task == "forecasting":
        features, target, timestamp_frame, feature_columns, history_tail = _build_forecasting_table(
            source_frame,
            target_column=target_column,
            time_column=time_column or "",
            feature_columns=requested_features,
            horizon=horizon,
        )
        split_index = max(1, int(len(features) * 0.8))
        train_x = features.iloc[:split_index].copy()
        val_x = features.iloc[split_index:].copy()
        train_y = target.iloc[:split_index].copy()
        val_y = target.iloc[split_index:].copy()
        val_time = timestamp_frame.iloc[split_index:].copy()
        schema_source = features
    else:
        feature_frame, target, feature_columns = _build_supervised_table(
            source_frame,
            target_column=target_column,
            feature_columns=requested_features,
            time_column=time_column,
        )
        schema_source = feature_frame.copy()
        model_frame = feature_frame.copy()
        model_frame[target_column] = target
        train_frame, val_frame = _train_val_split(
            model_frame,
            target_column=target_column,
            task=task,
            time_column=time_column if time_column in model_frame.columns else None,
        )
        train_x = train_frame.drop(columns=[target_column]).copy()
        val_x = val_frame.drop(columns=[target_column]).copy()
        train_y = train_frame[target_column].copy()
        val_y = val_frame[target_column].copy()
        if time_column and time_column in train_x.columns:
            train_x = _expand_time_features(train_x, time_column)
            val_x = _expand_time_features(val_x, time_column)
        feature_columns = list(train_x.columns)
        val_time = None
        history_tail = []

    if len(train_x) < 10 or len(val_x) < 3:
        raise ValueError("Dataset is too small after preprocessing. Provide more rows or simplify the task.")

    pipeline = _build_pipeline(train_x, "classification" if task == "classification" else "regression")
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(val_x)

    if task == "classification":
        labels = sorted(str(label) for label in pd.Series(train_y).astype(str).unique())
        probability_matrix = pipeline.predict_proba(val_x) if hasattr(pipeline, "predict_proba") else None
        probability_rows = []
        if probability_matrix is not None:
            for row in probability_matrix[:20]:
                probability_rows.append(
                    {
                        label: round(float(score), 4)
                        for label, score in zip(getattr(pipeline.named_steps["model"], "classes_", labels), row, strict=False)
                    }
                )
        evaluation = _classification_metrics(val_y.astype(str), predictions.astype(str), probability_matrix, labels)
        metrics = evaluation["metrics"]
        sample_predictions = []
        for index, (_, row) in enumerate(val_x.head(20).iterrows()):
            payload = {column: _json_value(value) for column, value in row.items()}
            entry = {
                "actual": str(val_y.iloc[index]),
                "predicted": str(predictions[index]),
                "features": payload,
            }
            if probability_rows:
                entry["probabilities"] = probability_rows[index]
            sample_predictions.append(entry)
        chart_data = []
        for index, predicted in enumerate(predictions[:40]):
            chart_data.append({"index": index + 1, "actual": str(val_y.iloc[index]), "predicted": str(predicted)})
    else:
        metrics = (_forecast_metrics if task == "forecasting" else _regression_metrics)(val_y.astype(float), predictions.astype(float))["metrics"]
        sample_predictions = []
        for index, (_, row) in enumerate(val_x.head(24).iterrows()):
            payload = {column: _json_value(value) for column, value in row.items()}
            item = {
                "actual": round(float(val_y.iloc[index]), 4),
                "predicted": round(float(predictions[index]), 4),
                "features": payload,
            }
            if val_time is not None:
                item["time"] = _json_value(val_time.iloc[index, 0])
            sample_predictions.append(item)
        chart_data = []
        for index, predicted in enumerate(predictions[:80]):
            point = {
                "index": index + 1,
                "actual": round(float(val_y.iloc[index]), 4),
                "predicted": round(float(predicted), 4),
            }
            if val_time is not None:
                point["time"] = _json_value(val_time.iloc[index, 0])
            chart_data.append(point)
        evaluation = {"metrics": metrics}

    feature_importance = _compute_feature_importance(
        pipeline,
        val_x.reset_index(drop=True),
        val_y.reset_index(drop=True),
        task="classification" if task == "classification" else "regression",
    )
    summary_text = _predictive_summary(task, metrics, feature_importance, target_column, horizon)
    forecast_preview = (
        _future_forecast(
            pipeline,
            source_frame,
            target_column=target_column,
            time_column=time_column or "",
            feature_columns=feature_columns,
            horizon=horizon,
        )
        if task == "forecasting"
        else []
    )

    input_schema = _feature_schema(schema_source, list(schema_source.columns))
    bundle = {
        "project_family": "predictive_ai",
        "task": task,
        "pipeline": pipeline,
        "metadata": {
            "target_column": target_column,
            "time_column": time_column,
            "feature_columns": feature_columns,
            "input_schema": input_schema,
            "forecast_horizon": horizon,
            "summary_text": summary_text,
        },
        "evaluation": {
            "task": task,
            "target_column": target_column,
            "time_column": time_column,
            "feature_columns": feature_columns,
            "dataset_rows": int(len(source_frame)),
            "dataset_columns": int(len(source_frame.columns)),
            "metrics": metrics,
            "summary_text": summary_text,
            "feature_importance": feature_importance,
            "sample_predictions": sample_predictions,
            "chart_data": chart_data,
            "input_schema": input_schema,
            "forecast_horizon": horizon,
            "history_tail": history_tail,
            "future_forecast": forecast_preview,
        },
    }
    if task == "classification":
        bundle["evaluation"]["labels"] = evaluation["labels"]
        bundle["evaluation"]["confusion_matrix"] = evaluation["confusion_matrix"]
    return bundle


def save_predictive_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_predictive_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_with_bundle(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    task = bundle["task"]
    pipeline: Pipeline = bundle["pipeline"]
    if task == "forecasting":
        return {
            "task": task,
            "summary": metadata.get("summary_text"),
            "forecast_horizon": metadata.get("forecast_horizon", 1),
            "future_forecast": bundle["evaluation"].get("future_forecast", []),
        }

    raw_frame = pd.DataFrame([payload])
    time_column = metadata.get("time_column")
    if time_column and time_column in raw_frame.columns:
        raw_frame = _expand_time_features(raw_frame, time_column)

    feature_columns = metadata.get("feature_columns", [])
    for column in feature_columns:
        if column not in raw_frame.columns:
            raw_frame[column] = np.nan
    raw_frame = raw_frame[feature_columns]

    if task == "classification":
        predicted = pipeline.predict(raw_frame)[0]
        response = {"task": task, "prediction": str(predicted), "summary": metadata.get("summary_text")}
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(raw_frame)[0]
            labels = getattr(pipeline.named_steps["model"], "classes_", [])
            response["probabilities"] = {
                str(label): round(float(score), 4)
                for label, score in zip(labels, probabilities, strict=False)
            }
        return response

    predicted_value = float(pipeline.predict(raw_frame)[0])
    return {
        "task": task,
        "prediction": round(predicted_value, 4),
        "summary": metadata.get("summary_text"),
    }
