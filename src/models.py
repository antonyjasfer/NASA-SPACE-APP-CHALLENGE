"""FLEX models, all validated with GroupKFold grouped by test_date (no row-level splits).

Run: python -m src.models
  a) 3-class outcome (Extinction / Disruption / Completion): multinomial logistic, then HGB.
  b) Disruption risk (Disruption vs not): logistic, then HGB.
  c) K (mm^2/s) regressor: Ridge, then HGB with monotonic-increasing x_o2 and po2_atm,
     plus HGB quantile (0.1 / 0.9) interval models.
  d) Nearest-real-test distance in standardized feature space.
Writes models/*.joblib and models/metrics.json.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import MODEL_FEATURES, MONOTONIC_INCREASING, build_features
from src.ingest import ROOT, load_clean

MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
N_SPLITS = 5
SEED = 42
CLASSES = ["Extinction", "Disruption", "Completion"]
GROUP_COL = "test_date"


# --------------------------------------------------------------------------- CV helpers
def group_splits(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold splits on test_date: a test date is never in both train and test."""
    groups = df[GROUP_COL].astype(str).to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(df, groups=groups))


@dataclass
class CVResult:
    per_fold: list[dict]
    oof_pred: np.ndarray
    oof_score: np.ndarray | None = None


def _summary(per_fold: list[dict]) -> dict:
    keys = [k for k in per_fold[0] if k not in ("fold", "n_test")]
    out = {}
    for k in keys:
        vals = np.array([f[k] for f in per_fold], dtype=float)
        out[k] = {"mean": _r(np.nanmean(vals)), "std": _r(np.nanstd(vals)),
                  "n_folds_defined": int(np.isfinite(vals).sum())}
    return out


def _r(x: Any) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


# --------------------------------------------------------------------------- estimators
def _linear_pre():
    return [SimpleImputer(strategy="median"), StandardScaler()]


def make_logistic():
    # lbfgs fits a multinomial model for >2 classes in sklearn >= 1.5.
    return make_pipeline(*_linear_pre(), LogisticRegression(
        max_iter=5000, class_weight="balanced", C=1.0))


def make_hgb_classifier():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def make_ridge():
    return make_pipeline(*_linear_pre(), Ridge(alpha=1.0))


def _hgb_reg_kwargs() -> dict:
    return dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=8,
                l2_regularization=1.0, random_state=SEED,
                monotonic_cst={f: 1 for f in MONOTONIC_INCREASING})


def make_hgb_regressor():
    return HistGradientBoostingRegressor(**_hgb_reg_kwargs())


def make_hgb_quantile(q: float):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **_hgb_reg_kwargs())


# --------------------------------------------------------------------------- CV loops
def cv_classifier(make: Callable, X: pd.DataFrame, y: pd.Series, splits, labels: list[str],
                  positive: str | None = None) -> CVResult:
    """Out-of-fold CV. For binary tasks pass `positive` to get ROC-AUC."""
    oof = np.empty(len(y), dtype=object)
    oof_score = np.full(len(y), np.nan)
    per_fold = []
    for i, (tr, te) in enumerate(splits):
        model = make().fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        oof[te] = pred
        row: dict[str, Any] = {"fold": i, "n_test": int(len(te))}
        yt = y.iloc[te]
        row["balanced_accuracy"] = _r(balanced_accuracy_score(yt, pred))
        if positive is None:
            row["macro_f1"] = _r(f1_score(yt, pred, labels=labels, average="macro", zero_division=0))
        else:
            pos_idx = list(model.classes_).index(positive)
            score = model.predict_proba(X.iloc[te])[:, pos_idx]
            oof_score[te] = score
            both = yt.nunique() == 2
            row["roc_auc"] = _r(roc_auc_score(yt == positive, score)) if both else None
        per_fold.append(row)
    return CVResult(per_fold, oof, oof_score if positive else None)


def cv_regressor(make: Callable, X: pd.DataFrame, y: pd.Series, splits) -> CVResult:
    oof = np.full(len(y), np.nan)
    per_fold = []
    for i, (tr, te) in enumerate(splits):
        pred = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        oof[te] = pred
        per_fold.append({"fold": i, "n_test": int(len(te)),
                         "r2": _r(r2_score(y.iloc[te], pred)),
                         "mae": _r(mean_absolute_error(y.iloc[te], pred))})
    return CVResult(per_fold, oof)


def fit_cqr(X: pd.DataFrame, y: pd.Series, groups: pd.Series, alpha: float = 0.2,
            cal_frac: float = 0.25, seed: int = SEED) -> dict:
    """Conformalized quantile regression with a group-aware calibration split: fit the
    0.1/0.9 HGB models on a seeded random ~75% of test dates, compute conformity scores on
    the held-out dates, and widen the band by their finite-sample (1 - alpha) quantile."""
    dates = groups.astype(str).to_numpy()
    uniq = np.unique(dates)
    rng = np.random.default_rng(seed)
    cal_dates = rng.choice(uniq, size=max(1, round(cal_frac * len(uniq))), replace=False)
    cal_mask = np.isin(dates, cal_dates)
    tr, cal = np.flatnonzero(~cal_mask), np.flatnonzero(cal_mask)
    lo_m = make_hgb_quantile(alpha / 2).fit(X.iloc[tr], y.iloc[tr])
    hi_m = make_hgb_quantile(1 - alpha / 2).fit(X.iloc[tr], y.iloc[tr])
    yc = y.iloc[cal].to_numpy()
    scores = np.maximum(lo_m.predict(X.iloc[cal]) - yc, yc - hi_m.predict(X.iloc[cal]))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return {"lo": lo_m, "hi": hi_m, "q_hat": float(np.quantile(scores, level, method="higher")),
            "alpha": alpha}


def predict_interval(cqr: dict, X: pd.DataFrame, conformal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    pad = cqr["q_hat"] if conformal else 0.0
    lo, hi = cqr["lo"].predict(X) - pad, cqr["hi"].predict(X) + pad
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return np.clip(lo, 0.0, None), hi  # K is a positive burning-rate constant


def cv_quantile_interval(X: pd.DataFrame, y: pd.Series, groups: pd.Series, splits) -> dict:
    """Out-of-fold coverage of the raw quantile band and of its conformalized version."""
    bands = {"raw_quantile": (np.full(len(y), np.nan), np.full(len(y), np.nan)),
             "conformalized": (np.full(len(y), np.nan), np.full(len(y), np.nan))}
    per_fold = []
    for i, (tr, te) in enumerate(splits):
        cqr = fit_cqr(X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        row: dict[str, Any] = {"fold": i, "q_hat": _r(cqr["q_hat"])}
        yt = y.iloc[te].to_numpy()
        for name, conformal in (("raw_quantile", False), ("conformalized", True)):
            lo, hi = predict_interval(cqr, X.iloc[te], conformal)
            bands[name][0][te], bands[name][1][te] = lo, hi
            row[f"{name}_coverage"] = _r(((yt >= lo) & (yt <= hi)).mean())
            row[f"{name}_mean_width"] = _r(np.mean(hi - lo))
        per_fold.append(row)
    pooled = {}
    for name, (lo, hi) in bands.items():
        pooled[name] = {"coverage": _r(((y.to_numpy() >= lo) & (y.to_numpy() <= hi)).mean()),
                        "mean_width_mm2_s": _r(np.mean(hi - lo))}
    return {"nominal_coverage": 0.8,
            "note": ("raw = HGB quantile 0.1/0.9 fitted on 75% of training dates; conformalized = same band "
                     "widened by split-conformal q_hat from the other 25% of training dates."),
            "per_fold": per_fold, "pooled_oof": pooled}


# --------------------------------------------------------------------------- nearest test
@dataclass
class NearestTestIndex:
    """Standardized reference matrix of real FLEX tests for "distance to nearest real test"."""
    features: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    matrix: np.ndarray
    flex_ids: np.ndarray
    test_nos: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str]) -> NearestTestIndex:
        raw = df[features].to_numpy(dtype=float)
        median = np.nanmedian(raw, axis=0)
        filled = np.where(np.isnan(raw), median, raw)
        mean, std = filled.mean(axis=0), filled.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return cls(features, median, mean, std, (filled - mean) / std,
                   df["flex_id"].to_numpy(), df["test_no"].to_numpy())

    def _transform(self, query: pd.DataFrame) -> np.ndarray:
        q = query[self.features].to_numpy(dtype=float)
        q = np.where(np.isnan(q), self.median, q)
        return (q - self.mean) / self.std

    def query(self, query: pd.DataFrame) -> pd.DataFrame:
        """Euclidean distance (in std units) to the nearest real test, plus its IDs."""
        z = self._transform(query)
        d = np.sqrt(((z[:, None, :] - self.matrix[None, :, :]) ** 2).sum(axis=2))
        idx = d.argmin(axis=1)
        return pd.DataFrame({"nearest_distance": d[np.arange(len(z)), idx],
                             "nearest_flex_id": self.flex_ids[idx],
                             "nearest_test_no": self.test_nos[idx]})


NEAREST_FEATURES = ["d0_mm", "x_o2", "p_atm", "x_n2", "x_co2", "x_he", "fuel_methanol"]


def nearest_test_distance(query: pd.DataFrame, index: NearestTestIndex | None = None) -> pd.DataFrame:
    """App helper: query needs NEAREST_FEATURES columns (see build_features)."""
    index = index or joblib.load(MODELS_DIR / "nearest_test.joblib")
    return index.query(query)


# --------------------------------------------------------------------------- pipeline
def _assert_no_shared_dates(df: pd.DataFrame, splits) -> None:
    for tr, te in splits:
        shared = set(df[GROUP_COL].iloc[tr]) & set(df[GROUP_COL].iloc[te])
        if shared:
            raise ValueError(f"test_date leak across folds: {shared}")


def _fold_dates(df, splits) -> list[int]:
    return [int(df[GROUP_COL].iloc[te].nunique()) for _, te in splits]


def run_outcome(df: pd.DataFrame) -> tuple[dict, dict]:
    X, y = df[MODEL_FEATURES], df["test_end"]
    splits = group_splits(df)
    _assert_no_shared_dates(df, splits)
    out: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name, make in [("logistic_multinomial", make_logistic),
                       ("hist_gradient_boosting", make_hgb_classifier)]:
        res = cv_classifier(make, X, y, splits, CLASSES)
        cm = confusion_matrix(y, res.oof_pred.astype(str), labels=CLASSES)
        out[name] = {
            "per_fold": res.per_fold, "mean": _summary(res.per_fold),
            "pooled_oof": {
                "macro_f1": _r(f1_score(y, res.oof_pred.astype(str), labels=CLASSES, average="macro")),
                "balanced_accuracy": _r(balanced_accuracy_score(y, res.oof_pred.astype(str))),
                "per_class_recall": {c: _r(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)},
            },
            "confusion_matrix_oof": {"labels_rows_true_cols_pred": CLASSES, "matrix": cm.tolist()},
        }
        fitted[f"outcome_{name}"] = make().fit(X, y)
    out["fold_n_test_dates"] = _fold_dates(df, splits)
    return out, fitted


def run_binary(df: pd.DataFrame, positive: str) -> tuple[dict, dict]:
    """`positive` vs the other two outcomes, e.g. Disruption risk or Extinction."""
    X = df[MODEL_FEATURES]
    negative = f"Not {positive.lower()}"
    y = df["test_end"].where(df["test_end"] == positive, negative)
    splits = group_splits(df)
    _assert_no_shared_dates(df, splits)
    out: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name, make in [("logistic", make_logistic), ("hist_gradient_boosting", make_hgb_classifier)]:
        res = cv_classifier(make, X, y, splits, [positive, negative], positive=positive)
        out[name] = {
            "per_fold": res.per_fold, "mean": _summary(res.per_fold),
            "pooled_oof": {"roc_auc": _r(roc_auc_score(y == positive, res.oof_score)),
                           "balanced_accuracy": _r(balanced_accuracy_score(y, res.oof_pred.astype(str)))},
        }
        fitted[f"{positive.lower()}_{name}"] = make().fit(X, y)
    return out, fitted


def run_k(df: pd.DataFrame) -> tuple[dict, dict]:
    sub = df[df["K_mm2_s"].notna()].reset_index(drop=True)
    X, y = sub[MODEL_FEATURES], sub["K_mm2_s"]
    splits = group_splits(sub)
    _assert_no_shared_dates(sub, splits)
    out: dict[str, Any] = {"n_rows_with_K": int(len(sub)), "fold_n_test_dates": _fold_dates(sub, splits)}
    fitted: dict[str, Any] = {}
    for name, make in [("ridge", make_ridge), ("hgb_monotonic", make_hgb_regressor)]:
        res = cv_regressor(make, X, y, splits)
        out[name] = {"per_fold": res.per_fold, "mean": _summary(res.per_fold),
                     "pooled_oof": {"r2": _r(r2_score(y, res.oof_pred)),
                                    "mae_mm2_s": _r(mean_absolute_error(y, res.oof_pred))}}
        fitted[f"k_{name}"] = make().fit(X, y)
    out["hgb_quantile_interval"] = cv_quantile_interval(X, y, sub[GROUP_COL], splits)
    fitted["k_interval_cqr"] = fit_cqr(X, y, sub[GROUP_COL])
    return out, fitted


def d2law_check(df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    for label, mask in [("all_tests", df.index == df.index), ("completion_only", df["test_end"] == "Completion")]:
        both = df.loc[mask, ["d2law_burn_time_s", "burn_time_s"]].dropna()
        ratio = both["d2law_burn_time_s"] / both["burn_time_s"]
        out[label] = {"n": int(len(both)),
                      "pearson_r": _r(both.corr().iloc[0, 1]) if len(both) > 2 else None,
                      "median_ratio_d2law_over_measured": _r(ratio.median())}
    out["note"] = ("d0^2/K is the time to burn the whole droplet. Extinction and Disruption end the burn "
                   "early, so for those tests the d^2-law time is an upper bound, not a prediction.")
    return out


def main() -> dict:
    df = build_features(load_clean())
    MODELS_DIR.mkdir(exist_ok=True)
    outcome, f1 = run_outcome(df)
    disruption, f2 = run_binary(df, "Disruption")
    extinction, f2b = run_binary(df, "Extinction")
    k_res, f3 = run_k(df)
    index = NearestTestIndex.fit(df, NEAREST_FEATURES)
    for name, model in {**f1, **f2, **f2b, **f3, "nearest_test": index}.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"source": "PSI-69 FLEX, DOI 10.60555/mbq8-0451", "n_rows": int(len(df)),
                 "n_test_dates": int(df[GROUP_COL].nunique()),
                 "outcome_counts": df["test_end"].value_counts().to_dict()},
        "validation": {"scheme": f"GroupKFold(n_splits={N_SPLITS}) grouped by test_date",
                       "row_level_random_split_used": False,
                       "sklearn_version": sklearn.__version__},
        "features": MODEL_FEATURES,
        "monotonic_increasing_for_K": list(MONOTONIC_INCREASING),
        "choices": {
            "disruption_kept": ("Disruption (droplet bursting) is kept as its own class, not dropped: "
                                "it dominates at O2 >= 0.25, which is the habitat-relevant regime. "
                                "Model (a) predicts all three outcomes; model (b) isolates Disruption risk."),
            "class_weight": "balanced (Completion has only 27 rows)",
            "benchmark_bar": ("CLAUDE.md's ~0.68 balanced acc / ~0.79 AUC bar refers to a binary extinction "
                              "model; compare it with extinction_binary, not with the 3-class model."),
            "excluded_from_features": ("d_ext_mm, K_mm2_s and burn_time_s are measured outcomes, not "
                                       "inputs, so they are never used to predict the test end."),
            "missing_values": ("kept as NaN in data; logistic/ridge use median imputation inside the CV "
                               "pipeline, HGB handles NaN natively."),
            "dup_id_193F001": "both rows kept (identical id, date and GMT); they share a date so always sit in the same fold.",
        },
        "outcome_3class": outcome,
        "disruption_binary": disruption,
        "extinction_binary": extinction,
        "K_regression": k_res,
        "d2law_consistency": d2law_check(df),
        "nearest_test": {"features": NEAREST_FEATURES,
                         "metric": "Euclidean distance on z-scored features (median-imputed)"},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: metrics[k] for k in ("data", "validation")}, indent=2, default=str))
    return metrics


if __name__ == "__main__":
    main()
