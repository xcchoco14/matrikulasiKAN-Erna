"""
KAN Baseline Multi-Dataset Evaluator  (with Optuna HPO)
=========================================================
Object-Oriented implementation of KAN (Kolmogorov-Arnold Networks) baseline
evaluation across 9 medical datasets with:
  - Optuna TPE hyperparameter optimisation (50 trials, seed=42)
  - Strict data-leakage prevention:
      • Hold-out test set is carved out FIRST and never touched during HPO/CV
      • Optuna tunes on an inner 3-fold CV of the training portion only
      • StandardScaler is fit inside every fold / every trial (no leakage)
  - 5-Fold Stratified Cross-Validation (outer loop, best params from HPO)
  - Per-fold detailed metrics (Accuracy, Precision, Recall, F1)
  - 95 % Confidence Intervals for all metrics (t-distribution)
  - Box-plot and line-chart visualisations per dataset
  - Summary comparison table across all datasets
  - JSON + CSV export of all results

Data-leakage prevention strategy
---------------------------------
  1. train_test_split (stratified, 20 %) → [train_full | holdout_test]
  2. Optuna objective uses inner 3-fold CV on train_full only:
       for each trial → inner split → scaler fit on inner-train → score on inner-val
  3. Outer 5-fold CV uses best Optuna params on train_full:
       for each fold → scaler fit on fold-train → evaluate on fold-val
  4. Final holdout evaluation:
       scaler fit on entire train_full → evaluate on holdout_test

Usage:
    python kan_baseline_optuna.py
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from scipy import stats
from ucimlrepo import fetch_ucirepo
from kan import KAN

warnings.filterwarnings("ignore")

# Silence Optuna's own verbose output; we print our own progress
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """Stores evaluation metrics for a single CV fold."""
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_loss: float
    test_loss: float


@dataclass
class ConfidenceInterval:
    """95 % confidence interval for a single metric."""
    mean: float
    lower: float
    upper: float
    std: float

    def __str__(self) -> str:
        half = (self.upper - self.lower) / 2
        return (
            f"{self.mean:.4f} ± {half:.4f}"
            f"  [{self.lower:.4f}, {self.upper:.4f}]"
        )


@dataclass
class DatasetResult:
    """Aggregated results for one dataset."""
    dataset_name: str
    n_samples: int
    n_features: int
    n_classes: int
    best_params: Dict = field(default_factory=dict)
    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    # Computed after all folds
    ci_accuracy: Optional[ConfidenceInterval] = None
    ci_precision: Optional[ConfidenceInterval] = None
    ci_recall: Optional[ConfidenceInterval] = None
    ci_f1: Optional[ConfidenceInterval] = None

    holdout_accuracy: float = 0.0
    holdout_precision: float = 0.0
    holdout_recall: float = 0.0
    holdout_f1: float = 0.0

    def compute_confidence_intervals(self, confidence: float = 0.95) -> None:
        """Compute t-distribution CIs (appropriate for n=5 folds)."""
        for metric in ("accuracy", "precision", "recall", "f1"):
            arr = np.array([getattr(fm, metric) for fm in self.fold_metrics])
            n = len(arr)
            mean = arr.mean()
            std = arr.std(ddof=1)
            t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
            margin = t_crit * (std / np.sqrt(n))
            setattr(self, f"ci_{metric}", ConfidenceInterval(
                mean=mean,
                lower=max(0.0, mean - margin),
                upper=min(1.0, mean + margin),
                std=std
            ))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

class BaseDatasetLoader(ABC):
    """Abstract base for all dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns (X, y, target_names)."""
        ...

    def _encode_and_clean(
        self,
        df: pd.DataFrame,
        target_col: str,
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        df = df.replace("?", np.nan)
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        df = df.dropna(subset=[target_col])
        for col in df.columns:
            if col != target_col and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        y_raw = df[target_col].values
        X = df.drop(columns=[target_col]).values.astype(float)
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw)
        return X, y


class BreastCancerLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Breast Cancer (sklearn)"

    def load(self):
        data = load_breast_cancer()
        return data.data, data.target, list(data.target_names)


class HepatitisLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Hepatitis"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/hepatitis/hepatitis.data")
        col_names = [
            "Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue",
            "Malaise", "Anorexia", "LiverBig", "LiverFirm", "SpleenPalpable",
            "Spiders", "Ascites", "Varices", "Bilirubin", "AlkPhosphate",
            "SGOT", "Albumin", "Protime", "Histology"
        ]
        df = pd.read_csv(url, names=col_names)
        X, y = self._encode_and_clean(df, target_col="Class")
        return X, y, ["Die", "Live"]


class LiverLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Liver (BUPA)"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/liver-disorders/bupa.data")
        col_names = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
        df = pd.read_csv(url, names=col_names)
        df["selector"] = df["selector"] - 1
        X = df.drop(columns=["selector"]).values.astype(float)
        y = df["selector"].values
        return X, y, ["Group 1", "Group 2"]


class ParkinsonsLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Parkinson's (Telemonitoring)"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/parkinsons/telemonitoring/parkinsons_updrs.data")
        df = pd.read_csv(url)
        target_col = "total_UPDRS"
        df["target"] = (df[target_col] >= df[target_col].median()).astype(int)
        drop_cols = ["subject#", "age", "sex", "test_time", "motor_UPDRS", target_col]
        X, y = self._encode_and_clean(df, target_col="target", drop_cols=drop_cols)
        return X, y, ["Mild", "Severe"]


class DiabetesLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Diabetes (Pima)"

    def load(self):
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master"
               "/pima-indians-diabetes.data.csv")
        col_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
        ]
        df = pd.read_csv(url, names=col_names)
        X = df.drop(columns=["Outcome"]).values.astype(float)
        y = df["Outcome"].values
        return X, y, ["Non-Diabetic", "Diabetic"]


class DermatologyLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Dermatology"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/dermatology/dermatology.data")
        df = pd.read_csv(url, header=None)
        df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["Class"]
        X, y = self._encode_and_clean(df, target_col="Class")
        return X, y, [f"Class {i+1}" for i in range(len(np.unique(y)))]


class WPBCLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "WPBC"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/breast-cancer-wisconsin/wpbc.data")
        df = pd.read_csv(url, header=None)
        df = df.drop(columns=[0])
        df.columns = ["Outcome"] + [f"f{i}" for i in range(df.shape[1] - 1)]
        X, y = self._encode_and_clean(df, target_col="Outcome")
        return X, y, ["Non-Recurrent", "Recurrent"]


class HeartDiseaseLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "Heart Disease (Cleveland)"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/heart-disease/processed.cleveland.data")
        col_names = [
            "Age", "Sex", "CP", "Trestbps", "Chol", "Fbs", "Restecg",
            "Thalach", "Exang", "Oldpeak", "Slope", "Ca", "Thal", "Target"
        ]
        df = pd.read_csv(url, names=col_names)
        df["Target"] = df["Target"].apply(
            lambda v: 0 if str(v).strip() == "0" else 1
        )
        X, y = self._encode_and_clean(df, target_col="Target")
        return X, y, ["No Disease", "Disease"]


class CTGLoader(BaseDatasetLoader):
    @property
    def name(self) -> str: return "CTG (Cardiotocography)"

    def load(self):
        ctg = fetch_ucirepo(id=193)
        X_raw = ctg.data.features
        y_raw = ctg.data.targets
        if isinstance(y_raw, pd.DataFrame):
            y_raw = y_raw.iloc[:, 0]
        X_df = X_raw.copy()
        for col in X_df.columns:
            if X_df[col].dtype == object:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
        X_df = X_df.fillna(X_df.median())
        X = X_df.values.astype(float)
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw.values.ravel())
        n_classes = len(np.unique(y))
        return X, y, [f"Class {i+1}" for i in range(n_classes)]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KAN TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class KANTrainer:
    """Wraps KAN model creation, training, and inference."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        params: Dict,
        seed: int = 42
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.params = params
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[KAN] = None
        self.loss_fn = nn.CrossEntropyLoss()

    def _build(self) -> KAN:
        return KAN(
            width=[self.num_features, self.params["n_hidden"], self.num_classes],
            grid=self.params["grid"],
            k=self.params["k"],
            seed=self.seed,
            device=self.device
        )

    def _tensors(self, X: np.ndarray, y: np.ndarray):
        return (
            torch.from_numpy(X).float().to(self.device),
            torch.from_numpy(y).long().to(self.device)
        )

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray
    ) -> Tuple[float, float]:
        """Train the model; return (train_loss, val_loss)."""
        self.model = self._build()
        X_tr_t, y_tr_t = self._tensors(X_train, y_train)
        X_va_t, y_va_t = self._tensors(X_val, y_val)

        self.model.fit(
            {
                "train_input": X_tr_t, "train_label": y_tr_t,
                "test_input": X_va_t,  "test_label": y_va_t
            },
            opt="LBFGS",
            steps=self.params["steps"],
            lr=self.params["lr"],
            loss_fn=self.loss_fn,
            lamb=self.params["lamb"]
        )

        with torch.no_grad():
            tr_loss = self.loss_fn(self.model(X_tr_t), y_tr_t).item()
            va_loss = self.loss_fn(self.model(X_va_t), y_va_t).item()
        return tr_loss, va_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(X_t), dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OPTUNA HPO  (leakage-free inner loop)
# ─────────────────────────────────────────────────────────────────────────────

class OptunaOptimizer:
    """
    Runs Optuna TPE search on an INNER cross-validation of the training set.

    Critical leakage-prevention properties:
    • Receives only X_train_full / y_train_full (hold-out test is never seen).
    • Each trial creates its own inner StratifiedKFold, fits a new scaler per
      inner fold, and evaluates on the inner validation split only.
    • The scaler from any trial is NEVER reused outside this class.
    """

    PARAM_SPACE = {
        "n_hidden": ("int",   2,    10),
        "grid":     ("int",   3,    9),
        "k":        ("int",   2,    4),
        "lr":       ("float", 1e-3, 0.1),
        "lamb":     ("float", 1e-4, 1e-1),
        "steps":    ("int",   10,   50),
    }

    def __init__(
        self,
        n_trials: int = 50,
        n_inner_folds: int = 3,
        random_state: int = 42
    ):
        self.n_trials = n_trials
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state

    def _make_objective(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        num_features: int,
        num_classes: int
    ):
        """Closure that creates the Optuna objective function."""
        avg = "binary" if num_classes == 2 else "weighted"
        inner_skf = StratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=self.random_state
        )

        def objective(trial: optuna.Trial) -> float:
            # ── Suggest hyperparameters ────────────────────────────────
            params = {
                "n_hidden": trial.suggest_int("n_hidden", 2, 10),
                "grid":     trial.suggest_int("grid",     3, 9),
                "k":        trial.suggest_int("k",        2, 4),
                "lr":       trial.suggest_float("lr",     1e-3, 0.1,  log=True),
                "lamb":     trial.suggest_float("lamb",   1e-4, 1e-1, log=True),
                "steps":    trial.suggest_int("steps",   10, 50),
            }

            fold_f1s = []
            for fold_i, (tr_idx, va_idx) in enumerate(
                inner_skf.split(X_train_full, y_train_full)
            ):
                X_i_tr, X_i_va = X_train_full[tr_idx], X_train_full[va_idx]
                y_i_tr, y_i_va = y_train_full[tr_idx], y_train_full[va_idx]

                # ── Scaler fit ONLY on inner-train, never on inner-val ──
                scaler = StandardScaler()
                X_i_tr_s = scaler.fit_transform(X_i_tr)
                X_i_va_s = scaler.transform(X_i_va)

                trainer = KANTrainer(
                    num_features=num_features,
                    num_classes=num_classes,
                    params=params,
                    seed=self.random_state + fold_i
                )
                try:
                    trainer.fit(X_i_tr_s, y_i_tr, X_i_va_s, y_i_va)
                    preds = trainer.predict(X_i_va_s)
                    fold_f1s.append(
                        f1_score(y_i_va, preds, average=avg, zero_division=0)
                    )
                except Exception:
                    # Prune bad trials instead of crashing
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_f1s))

        return objective

    def optimize(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        num_features: int,
        num_classes: int,
        verbose: bool = True
    ) -> Dict:
        """
        Run Optuna study and return the best hyperparameter dict.
        X_train_full / y_train_full must NOT include the hold-out test set.
        """
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        objective = self._make_objective(
            X_train_full, y_train_full, num_features, num_classes
        )

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        best_val = study.best_value

        if verbose:
            print(
                f"    [Optuna] Best inner-CV F1 = {best_val:.4f}  |  "
                f"Params: {best}"
            )

        return best


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-VALIDATION ENGINE  (outer loop, uses best params)
# ─────────────────────────────────────────────────────────────────────────────

class CrossValidator:
    """
    Outer 5-fold stratified CV.
    Scaler is fit inside each fold (no leakage).
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        num_classes: int,
        best_params: Dict,
        verbose: bool = True
    ) -> List[FoldMetrics]:
        fold_metrics: List[FoldMetrics] = []
        avg = "binary" if num_classes == 2 else "weighted"

        for fold_idx, (train_idx, val_idx) in enumerate(
            self.skf.split(X, y), start=1
        ):
            if verbose:
                print(f"    Fold {fold_idx}/{self.n_splits} ...", end=" ", flush=True)

            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # ── Scaler fit only on fold-train ──────────────────────────
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            trainer = KANTrainer(
                num_features=num_features,
                num_classes=num_classes,
                params=best_params,
                seed=self.random_state + fold_idx
            )
            tr_loss, va_loss = trainer.fit(X_tr_s, y_tr, X_va_s, y_va)
            preds = trainer.predict(X_va_s)

            fm = FoldMetrics(
                fold=fold_idx,
                accuracy=accuracy_score(y_va, preds),
                precision=precision_score(y_va, preds, average=avg, zero_division=0),
                recall=recall_score(y_va, preds, average=avg, zero_division=0),
                f1=f1_score(y_va, preds, average=avg, zero_division=0),
                train_loss=tr_loss,
                test_loss=va_loss
            )
            fold_metrics.append(fm)

            if verbose:
                print(
                    f"Acc={fm.accuracy:.4f} | Prec={fm.precision:.4f} | "
                    f"Rec={fm.recall:.4f} | F1={fm.f1:.4f}"
                )

        return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

class ResultVisualizer:
    """Creates and saves all visualisations."""

    METRICS = ["accuracy", "precision", "recall", "f1"]
    COLORS  = {
        "accuracy":  "#2563EB",
        "precision": "#16A34A",
        "recall":    "#D97706",
        "f1":        "#9333EA"
    }

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _safe(self, name: str) -> str:
        return name.replace(" ", "_").replace("/", "-") \
                   .replace("(", "").replace(")", "")

    # ── per-dataset box plot ────────────────────────────────────────────────

    def plot_fold_boxplot(self, result: DatasetResult) -> str:
        data = {m: [getattr(fm, m) for fm in result.fold_metrics]
                for m in self.METRICS}

        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
        fig.suptitle(
            f"Per-Fold Metrics — {result.dataset_name}\n"
            f"(5-Fold CV, n={result.n_samples})",
            fontsize=13, fontweight="bold", y=1.01
        )

        for ax, metric in zip(axes, self.METRICS):
            values = data[metric]
            ci = getattr(result, f"ci_{metric}")

            bp = ax.boxplot(
                values, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=self.COLORS[metric], linewidth=1.5),
                capprops=dict(color=self.COLORS[metric], linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor=self.COLORS[metric],
                                markersize=5)
            )
            bp["boxes"][0].set_facecolor(self.COLORS[metric])
            bp["boxes"][0].set_alpha(0.75)

            x_jitter = np.random.uniform(-0.08, 0.08, len(values))
            for j, v in enumerate(values):
                ax.scatter(1 + x_jitter[j], v, color=self.COLORS[metric],
                           zorder=5, s=40, edgecolors="white", linewidths=0.7)
                ax.annotate(f"F{j+1}", (1 + x_jitter[j], v),
                            textcoords="offset points", xytext=(5, 0),
                            fontsize=7, color="#555555")

            ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_ylim(max(0, min(values) - 0.08), min(1.0, max(values) + 0.08))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.set_xlabel(
                f"Mean={ci.mean:.4f}\n95% CI [{ci.lower:.4f}, {ci.upper:.4f}]",
                fontsize=8
            )
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{self._safe(result.dataset_name)}_fold_boxplot.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── per-dataset line chart ──────────────────────────────────────────────

    def plot_fold_line(self, result: DatasetResult) -> str:
        folds = [fm.fold for fm in result.fold_metrics]
        fig, ax = plt.subplots(figsize=(8, 4))

        for metric in self.METRICS:
            values = [getattr(fm, metric) for fm in result.fold_metrics]
            ax.plot(folds, values, marker="o", label=metric.capitalize(),
                    color=self.COLORS[metric], linewidth=1.8, markersize=6)

        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Per-Fold Metrics Trend — {result.dataset_name}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{self._safe(result.dataset_name)}_fold_line.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── confusion matrix ────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        result: DatasetResult,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: List[str]
    ) -> str:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(6, len(target_names) * 1.5),
                                        max(5, len(target_names) * 1.3)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title(
            f"Confusion Matrix — {result.dataset_name}\n"
            f"Holdout  Acc={result.holdout_accuracy:.4f}",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        path = os.path.join(self.save_dir,
                            f"{self._safe(result.dataset_name)}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Optuna optimisation history ─────────────────────────────────────────

    def plot_optuna_history(
        self,
        result: DatasetResult,
        study: optuna.Study
    ) -> str:
        trials = [t for t in study.trials
                  if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            return ""

        trial_nums = [t.number for t in trials]
        values = [t.value for t in trials]
        best_so_far = [max(values[:i+1]) for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(trial_nums, values, alpha=0.5, s=20,
                   color="#6B7280", label="Trial F1")
        ax.plot(trial_nums, best_so_far, color="#DC2626",
                linewidth=2, label="Best so far")
        ax.set_xlabel("Trial number", fontsize=10)
        ax.set_ylabel("Inner-CV F1 (mean)", fontsize=10)
        ax.set_title(
            f"Optuna Optimisation History — {result.dataset_name}\n"
            f"Best F1={max(values):.4f}  |  Best params: {result.best_params}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(self.save_dir,
                            f"{self._safe(result.dataset_name)}_optuna_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── cross-dataset summary ───────────────────────────────────────────────

    def plot_summary_comparison(self, all_results: List[DatasetResult]) -> str:
        n = len(all_results)
        x = np.arange(n)
        width = 0.2
        labels = [r.dataset_name.split("(")[0].strip() for r in all_results]

        fig, ax = plt.subplots(figsize=(max(14, n * 1.6), 6))

        for i, metric in enumerate(self.METRICS):
            means  = [getattr(r, f"ci_{metric}").mean  for r in all_results]
            lowers = [getattr(r, f"ci_{metric}").lower for r in all_results]
            uppers = [getattr(r, f"ci_{metric}").upper for r in all_results]
            errs = [
                [m - lo for m, lo in zip(means, lowers)],
                [hi - m  for m, hi in zip(means, uppers)]
            ]
            ax.bar(
                x + i * width, means, width,
                label=metric.capitalize(),
                color=self.COLORS[metric], alpha=0.8,
                yerr=errs, capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#333333"}
            )

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score (95 % CI)", fontsize=11)
        ax.set_title(
            "KAN + Optuna Baseline — All Datasets (5-Fold CV, 95 % CI)",
            fontsize=13, fontweight="bold"
        )
        ax.set_ylim(0, 1.12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(self.save_dir, "summary_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULT EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

class ResultExporter:
    """Serialises DatasetResult objects to JSON and CSV."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def export_json(self, all_results: List[DatasetResult]) -> str:
        payload = []
        for r in all_results:
            payload.append({
                "dataset": r.dataset_name,
                "n_samples": r.n_samples,
                "n_features": r.n_features,
                "n_classes": r.n_classes,
                "best_optuna_params": r.best_params,
                "cross_validation": {
                    "folds": [
                        {
                            "fold": fm.fold,
                            "accuracy":   fm.accuracy,
                            "precision":  fm.precision,
                            "recall":     fm.recall,
                            "f1":         fm.f1,
                            "train_loss": fm.train_loss,
                            "test_loss":  fm.test_loss
                        }
                        for fm in r.fold_metrics
                    ],
                    "confidence_intervals_95": {
                        m: {
                            "mean":  getattr(r, f"ci_{m}").mean,
                            "lower": getattr(r, f"ci_{m}").lower,
                            "upper": getattr(r, f"ci_{m}").upper,
                            "std":   getattr(r, f"ci_{m}").std
                        }
                        for m in ("accuracy", "precision", "recall", "f1")
                    }
                },
                "holdout_test": {
                    "accuracy":  r.holdout_accuracy,
                    "precision": r.holdout_precision,
                    "recall":    r.holdout_recall,
                    "f1":        r.holdout_f1
                }
            })

        path = os.path.join(self.save_dir, "all_results.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        return path

    def export_csv_summary(self, all_results: List[DatasetResult]) -> str:
        rows = []
        for r in all_results:
            row = {
                "Dataset":    r.dataset_name,
                "N_Samples":  r.n_samples,
                "N_Features": r.n_features,
                "N_Classes":  r.n_classes,
            }
            # Best params columns
            for k, v in r.best_params.items():
                row[f"opt_{k}"] = v
            # CV CI columns
            for m in ("accuracy", "precision", "recall", "f1"):
                ci = getattr(r, f"ci_{m}")
                half = (ci.upper - ci.lower) / 2
                row[f"CV_{m}_mean"]       = round(ci.mean,  4)
                row[f"CV_{m}_CI95_lower"] = round(ci.lower, 4)
                row[f"CV_{m}_CI95_upper"] = round(ci.upper, 4)
                row[f"CV_{m}_pm"]         = round(half,     4)
                row[f"CV_{m}_std"]        = round(ci.std,   4)
            row["Holdout_Accuracy"]  = round(r.holdout_accuracy,  4)
            row["Holdout_Precision"] = round(r.holdout_precision, 4)
            row["Holdout_Recall"]    = round(r.holdout_recall,    4)
            row["Holdout_F1"]        = round(r.holdout_f1,        4)
            rows.append(row)

        path = os.path.join(self.save_dir, "summary_table.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_csv_per_fold(self, all_results: List[DatasetResult]) -> str:
        rows = []
        for r in all_results:
            for fm in r.fold_metrics:
                rows.append({
                    "Dataset":    r.dataset_name,
                    "Fold":       fm.fold,
                    "Accuracy":   round(fm.accuracy,   4),
                    "Precision":  round(fm.precision,  4),
                    "Recall":     round(fm.recall,     4),
                    "F1":         round(fm.f1,         4),
                    "Train_Loss": round(fm.train_loss, 4),
                    "Test_Loss":  round(fm.test_loss,  4)
                })

        path = os.path.join(self.save_dir, "per_fold_metrics.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class KANBaselinePipeline:
    """
    Main orchestrator — leakage-free three-stage pipeline per dataset:

    Stage A — Data split (done FIRST, ONCE):
        train_test_split → [train_full (80%) | holdout_test (20%)]
        holdout_test is sealed until Stage C.

    Stage B — Hyperparameter search (on train_full only):
        OptunaOptimizer → inner 3-fold CV of train_full
        Scaler: fit inside every inner fold; never touches holdout_test.

    Stage C — Outer evaluation (on train_full, best params fixed):
        CrossValidator → outer 5-fold CV of train_full
        Scaler: fit inside every outer fold; never touches holdout_test.
        Final model: scaler fit on entire train_full → evaluate on holdout_test.
    """

    def __init__(
        self,
        save_dir:      str   = "./kan_baseline_optuna_results",
        n_folds:       int   = 5,
        holdout_size:  float = 0.2,
        n_trials:      int   = 50,
        n_inner_folds: int   = 3,
        random_state:  int   = 42,
        verbose:       bool  = True
    ):
        self.save_dir     = save_dir
        self.n_folds      = n_folds
        self.holdout_size = holdout_size
        self.n_trials     = n_trials
        self.random_state = random_state
        self.verbose      = verbose

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = OptunaOptimizer(
            n_trials=n_trials,
            n_inner_folds=n_inner_folds,
            random_state=random_state
        )
        self.cv        = CrossValidator(n_splits=n_folds, random_state=random_state)
        self.viz       = ResultVisualizer(save_dir)
        self.exporter  = ResultExporter(save_dir)

        self.loaders: List[BaseDatasetLoader] = [
            BreastCancerLoader(),
            HepatitisLoader(),
            LiverLoader(),
            ParkinsonsLoader(),
            DiabetesLoader(),
            DermatologyLoader(),
            WPBCLoader(),
            HeartDiseaseLoader(),
            CTGLoader(),
        ]

        self.all_results: List[DatasetResult] = []

    # ── helpers ─────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _sep(self, char: str = "─", n: int = 72) -> None:
        self._log(char * n)

    # ── per-dataset pipeline ─────────────────────────────────────────────────

    def _process_dataset(self, loader: BaseDatasetLoader) -> Optional[DatasetResult]:
        self._sep()
        self._log(f"  Dataset : {loader.name}")

        # ── Load ────────────────────────────────────────────────────────────
        try:
            X, y, target_names = loader.load()
        except Exception as exc:
            self._log(f"  [ERROR] Failed to load {loader.name}: {exc}")
            return None

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self._log(f"  Samples={n_samples}  Features={n_features}  Classes={n_classes}")

        result = DatasetResult(
            dataset_name=loader.name,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes
        )

        # ── STAGE A: Hold-out carve-out (holdout_test is SEALED until Stage C) ──
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=self.holdout_size,
            random_state=self.random_state,
            stratify=y
        )

        # ── STAGE B: Optuna HPO (inner CV on train_full only) ───────────────
        self._log(
            f"\n  [Stage B] Optuna HPO ({self.n_trials} trials, "
            f"{self.optimizer.n_inner_folds}-fold inner CV) …"
        )
        # Rebuild study so we can pass it to the visualizer later
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study   = optuna.create_study(direction="maximize", sampler=sampler)

        objective = self.optimizer._make_objective(
            X_train_full, y_train_full, n_features, n_classes
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params  = study.best_params
        best_inner_f1 = study.best_value
        result.best_params = best_params

        self._log(
            f"    Best inner-CV F1 = {best_inner_f1:.4f}\n"
            f"    Best params      = {best_params}"
        )

        # ── STAGE C-i: Outer 5-fold CV (train_full, best_params fixed) ──────
        self._log(f"\n  [Stage C-i] Outer {self.n_folds}-Fold CV (best params) …")
        fold_metrics = self.cv.run(
            X_train_full, y_train_full,
            num_features=n_features,
            num_classes=n_classes,
            best_params=best_params,
            verbose=self.verbose
        )
        result.fold_metrics = fold_metrics
        result.compute_confidence_intervals()

        # ── STAGE C-ii: Final holdout evaluation ────────────────────────────
        self._log("\n  [Stage C-ii] Final model → holdout test …")

        # Scaler fit on ALL of train_full (clean)
        scaler_final = StandardScaler()
        X_tr_s = scaler_final.fit_transform(X_train_full)
        X_te_s = scaler_final.transform(X_test)   # transform only, no fit

        final_trainer = KANTrainer(
            num_features=n_features,
            num_classes=n_classes,
            params=best_params,
            seed=self.random_state
        )
        final_trainer.fit(X_tr_s, y_train_full, X_te_s, y_test)
        test_preds = final_trainer.predict(X_te_s)

        avg = "binary" if n_classes == 2 else "weighted"
        result.holdout_accuracy  = accuracy_score(y_test, test_preds)
        result.holdout_precision = precision_score(y_test, test_preds, average=avg, zero_division=0)
        result.holdout_recall    = recall_score(y_test, test_preds, average=avg, zero_division=0)
        result.holdout_f1        = f1_score(y_test, test_preds, average=avg, zero_division=0)

        self._log(
            f"    [HOLDOUT] Acc={result.holdout_accuracy:.4f} | "
            f"Prec={result.holdout_precision:.4f} | "
            f"Rec={result.holdout_recall:.4f} | "
            f"F1={result.holdout_f1:.4f}"
        )
        self._log(
            "\n  Classification Report (Holdout):\n"
            + classification_report(y_test, test_preds,
                                    target_names=target_names, zero_division=0)
        )

        # CI summary
        self._log("  95 % Confidence Intervals (outer CV):")
        for m in ("accuracy", "precision", "recall", "f1"):
            ci = getattr(result, f"ci_{m}")
            self._log(f"    {m.capitalize():12s}: {ci}")

        # ── Visualisations ───────────────────────────────────────────────────
        self._log("\n  Generating plots …")
        self.viz.plot_fold_boxplot(result)
        self.viz.plot_fold_line(result)
        self.viz.plot_confusion_matrix(result, y_test, test_preds, target_names)
        self.viz.plot_optuna_history(result, study)

        return result

    # ── main entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        self._log("\n" + "=" * 72)
        self._log("  KAN Baseline + Optuna HPO  — Multi-Dataset Evaluator")
        self._log(f"  Device   : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self._log(f"  Trials   : {self.n_trials}  |  Outer folds : {self.n_folds}")
        self._log(f"  Holdout  : {int(self.holdout_size * 100)} %  |  Seed : {self.random_state}")
        self._log("=" * 72)

        for loader in self.loaders:
            result = self._process_dataset(loader)
            if result is not None:
                self.all_results.append(result)

        if self.all_results:
            self._sep()
            self._log("\n  Generating summary comparison chart …")
            self.viz.plot_summary_comparison(self.all_results)

            self._log("  Exporting results …")
            p_json     = self.exporter.export_json(self.all_results)
            p_csv      = self.exporter.export_csv_summary(self.all_results)
            p_fold_csv = self.exporter.export_csv_per_fold(self.all_results)

            self._log(f"    JSON  → {p_json}")
            self._log(f"    CSV   → {p_csv}")
            self._log(f"    Folds → {p_fold_csv}")

            self._print_summary_table()

        self._log("\n  Done.  Results saved to: " + self.save_dir)

    def _print_summary_table(self) -> None:
        self._sep()
        self._log("FINAL SUMMARY  (outer 5-Fold CV — 95 % Confidence Intervals)")
        self._sep()
        header = (
            f"{'Dataset':<35} {'Acc (mean ± CI)':<22} "
            f"{'Prec (mean ± CI)':<22} {'Rec (mean ± CI)':<22} "
            f"{'F1 (mean ± CI)':<22}"
        )
        self._log(header)
        self._log("-" * len(header))

        for r in self.all_results:
            cols = []
            for m in ("accuracy", "precision", "recall", "f1"):
                ci   = getattr(r, f"ci_{m}")
                half = (ci.upper - ci.lower) / 2
                cols.append(f"{ci.mean:.4f} ± {half:.4f}")
            row = (
                f"{r.dataset_name:<35} {cols[0]:<22} {cols[1]:<22} "
                f"{cols[2]:<22} {cols[3]:<22}"
            )
            self._log(row)

        self._log("-" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Google Colab Drive integration (optional)
    try:
        from google.colab import drive as _drive
        _drive.mount("/content/drive")
        SAVE_DIR = (
            "/content/drive/MyDrive/Colab_Notebooks/medical"
            "/kan_baseline_optuna_results"
        )
    except (ImportError, Exception):
        SAVE_DIR = "./kan_baseline_optuna_results"

    pipeline = KANBaselinePipeline(
        save_dir      = SAVE_DIR,
        n_folds       = 5,       # outer CV folds
        holdout_size  = 0.2,     # 20 % sealed test set
        n_trials      = 50,      # Optuna trials per dataset
        n_inner_folds = 3,       # inner CV folds inside Optuna objective
        random_state  = 42,
        verbose       = True
    )
    pipeline.run()
