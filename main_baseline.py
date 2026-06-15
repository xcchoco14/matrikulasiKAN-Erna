"""
KAN Baseline Multi-Dataset Evaluator
=====================================
Object-Oriented implementation of KAN (Kolmogorov-Arnold Networks) baseline
evaluation across 9 medical datasets with:
  - 5-Fold Stratified Cross-Validation
  - Per-fold detailed metrics (Accuracy, Precision, Recall, F1)
  - 95% Confidence Intervals for all metrics
  - Box-plot visualizations per dataset
  - Summary comparison table across all datasets
  - JSON + CSV export of all results

Revision-compliant based on professor's feedback:
  1. Per-fold metrics are logged and visualized (box plots)
  2. 95% Confidence Intervals reported for all key metrics
  3. Single OOP file structure

Usage:
    python kan_baseline_multi_dataset.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA CLASSES  (structured result containers)
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
        return f"{self.mean:.4f} ± {(self.upper - self.lower) / 2:.4f}  [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class DatasetResult:
    """Aggregated results for one dataset."""
    dataset_name: str
    n_samples: int
    n_features: int
    n_classes: int
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
        """Compute CIs using the t-distribution for small samples."""
        metrics_map = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        for fm in self.fold_metrics:
            metrics_map["accuracy"].append(fm.accuracy)
            metrics_map["precision"].append(fm.precision)
            metrics_map["recall"].append(fm.recall)
            metrics_map["f1"].append(fm.f1)

        for key, values in metrics_map.items():
            arr = np.array(values)
            n = len(arr)
            mean = arr.mean()
            std = arr.std(ddof=1)
            # t-distribution CI
            t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
            margin = t_crit * (std / np.sqrt(n))
            ci = ConfidenceInterval(
                mean=mean,
                lower=max(0.0, mean - margin),
                upper=min(1.0, mean + margin),
                std=std
            )
            setattr(self, f"ci_{key}", ci)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET LOADERS  (one class per dataset, common interface)
# ─────────────────────────────────────────────────────────────────────────────

class BaseDatasetLoader(ABC):
    """Abstract base for all dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

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
        """Common preprocessing: drop columns, encode categoricals, handle missing."""
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        # Replace known missing value markers
        df = df.replace("?", np.nan)

        # Encode categorical feature columns
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Drop rows where target is missing
        df = df.dropna(subset=[target_col])

        # Fill remaining feature NaNs with column median
        for col in df.columns:
            if col != target_col and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        y_raw = df[target_col].values
        X = df.drop(columns=[target_col]).values.astype(float)

        # Encode target to 0-based integers
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw)
        return X, y


class BreastCancerLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Breast Cancer (sklearn)"

    def load(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target
        target_names = list(data.target_names)
        return X, y, target_names


class HepatitisLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Hepatitis"

    def load(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
        col_names = [
            "Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue",
            "Malaise", "Anorexia", "LiverBig", "LiverFirm", "SpleenPalpable",
            "Spiders", "Ascites", "Varices", "Bilirubin", "AlkPhosphate",
            "SGOT", "Albumin", "Protime", "Histology"
        ]
        df = pd.read_csv(url, names=col_names)
        X, y = self._encode_and_clean(df, target_col="Class")
        target_names = ["Die", "Live"]
        return X, y, target_names


class LiverLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Liver (BUPA)"

    def load(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
        col_names = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
        df = pd.read_csv(url, names=col_names)
        df["selector"] = df["selector"] - 1
        X = df.drop(columns=["selector"]).values.astype(float)
        y = df["selector"].values
        target_names = ["Group 1", "Group 2"]
        return X, y, target_names



class ParkinsonsLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Parkinson's (Telemonitoring)"

    def load(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
        df = pd.read_csv(url)

        # Binarize total_UPDRS: ≥ median → 1 (severe), else 0 (mild)
        target_col = "total_UPDRS"
        median_val = df[target_col].median()
        df["target"] = (df[target_col] >= median_val).astype(int)

        drop_cols = ["subject#", "age", "sex", "test_time", "motor_UPDRS", target_col]
        X, y = self._encode_and_clean(df, target_col="target", drop_cols=drop_cols)
        target_names = ["Mild", "Severe"]
        return X, y, target_names


class DiabetesLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Diabetes (Pima)"

    def load(self):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        col_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
        ]
        df = pd.read_csv(url, names=col_names)
        X = df.drop(columns=["Outcome"]).values.astype(float)
        y = df["Outcome"].values
        target_names = ["Non-Diabetic", "Diabetic"]
        return X, y, target_names


class DermatologyLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Dermatology"

    def load(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
        df = pd.read_csv(url, header=None)
        # Last column is the target (1-6)
        df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["Class"]
        X, y = self._encode_and_clean(df, target_col="Class")
        target_names = [f"Class {i+1}" for i in range(len(np.unique(y)))]
        return X, y, target_names


class WPBCLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "WPBC"

    def load(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"
        df = pd.read_csv(url, header=None)
        # Col 0 = ID, Col 1 = outcome (N/R), rest = features
        df = df.drop(columns=[0])  # drop ID
        df.columns = ["Outcome"] + [f"f{i}" for i in range(df.shape[1] - 1)]
        X, y = self._encode_and_clean(df, target_col="Outcome")
        target_names = ["Non-Recurrent", "Recurrent"]
        return X, y, target_names


class HeartDiseaseLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Heart Disease (Cleveland)"

    def load(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        col_names = [
            "Age", "Sex", "CP", "Trestbps", "Chol", "Fbs", "Restecg",
            "Thalach", "Exang", "Oldpeak", "Slope", "Ca", "Thal", "Target"
        ]
        df = pd.read_csv(url, names=col_names)
        # Binarize: 0 = no disease, 1 = disease (original 1-4)
        df["Target"] = (df["Target"].apply(
            lambda v: 0 if str(v).strip() == "0" else 1)
        )
        X, y = self._encode_and_clean(df, target_col="Target")
        target_names = ["No Disease", "Disease"]
        return X, y, target_names


class CTGLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "CTG (Cardiotocography)"

    def load(self):
        cardiotocography = fetch_ucirepo(id=193)
        X_raw = cardiotocography.data.features
        y_raw = cardiotocography.data.targets

        if isinstance(y_raw, pd.DataFrame):
            y_raw = y_raw.iloc[:, 0]

        # Handle NaN
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
        target_names = [f"Class {i+1}" for i in range(n_classes)]
        return X, y, target_names


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KAN TRAINER  (encapsulates model init + training + inference)
# ─────────────────────────────────────────────────────────────────────────────

class KANTrainer:
    """Wraps KAN model creation, training, and inference."""

    DEFAULT_PARAMS = {
        "n_hidden": 5,
        "grid": 5,
        "k": 4,
        "lr": 0.01,
        "lamb": 0.01,
        "steps": 50
    }

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        params: Optional[Dict] = None,
        seed: int = 42
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[KAN] = None
        self.loss_fn = nn.CrossEntropyLoss()

    def _build_model(self) -> KAN:
        return KAN(
            width=[self.num_features, self.params["n_hidden"], self.num_classes],
            grid=self.params["grid"],
            k=self.params["k"],
            seed=self.seed,
            device=self.device
        )

    def _to_tensor(self, X: np.ndarray, y: np.ndarray):
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        return X_t, y_t

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[float, float]:
        """Train on (X_train, y_train), evaluate on (X_val, y_val).
        Returns (train_loss, val_loss)."""
        self.model = self._build_model()

        X_tr_t, y_tr_t = self._to_tensor(X_train, y_train)
        X_va_t, y_va_t = self._to_tensor(X_val, y_val)

        self.model.fit(
            {
                "train_input": X_tr_t,
                "train_label": y_tr_t,
                "test_input": X_va_t,
                "test_label": y_va_t
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
        """Return class predictions."""
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        with torch.no_grad():
            return self.loss_fn(self.model(X_t), y_t).item()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CrossValidator:
    """Runs stratified k-fold CV and returns per-fold FoldMetrics."""

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
        kan_params: Optional[Dict] = None,
        verbose: bool = True
    ) -> List[FoldMetrics]:
        """Execute CV and return list of per-fold metrics."""
        fold_metrics: List[FoldMetrics] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y), start=1):
            if verbose:
                print(f"    Fold {fold_idx}/{self.n_splits} ...", end=" ", flush=True)

            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # Scale within fold (no leakage)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            trainer = KANTrainer(
                num_features=num_features,
                num_classes=num_classes,
                params=kan_params,
                seed=self.random_state + fold_idx
            )
            tr_loss, va_loss = trainer.fit(X_tr_s, y_tr, X_va_s, y_va)
            preds = trainer.predict(X_va_s)

            avg = "binary" if num_classes == 2 else "weighted"
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
# 5.  VISUALIZER  (all plot generation)
# ─────────────────────────────────────────────────────────────────────────────

class ResultVisualizer:
    """Creates and saves all required visualizations."""

    METRICS = ["accuracy", "precision", "recall", "f1"]
    COLORS = {
        "accuracy": "#2563EB",
        "precision": "#16A34A",
        "recall": "#D97706",
        "f1": "#9333EA"
    }

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ── per-dataset box plot ────────────────────────────────────────────────

    def plot_fold_boxplot(self, result: DatasetResult) -> str:
        """Box plots of per-fold metrics for one dataset."""
        data = {
            m: [getattr(fm, m) for fm in result.fold_metrics]
            for m in self.METRICS
        }

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
                values,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=self.COLORS[metric], linewidth=1.5),
                capprops=dict(color=self.COLORS[metric], linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor=self.COLORS[metric], markersize=5)
            )
            bp["boxes"][0].set_facecolor(self.COLORS[metric])
            bp["boxes"][0].set_alpha(0.75)

            # Scatter individual fold points
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
        safe_name = result.dataset_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        path = os.path.join(self.save_dir, f"{safe_name}_fold_boxplot.png")
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
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=target_names, yticklabels=target_names, ax=ax
        )
        ax.set_title(
            f"Confusion Matrix — {result.dataset_name}\n"
            f"Holdout Test  Acc={result.holdout_accuracy:.4f}",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        safe_name = result.dataset_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        path = os.path.join(self.save_dir, f"{safe_name}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── cross-dataset comparison ────────────────────────────────────────────

    def plot_summary_comparison(self, all_results: List[DatasetResult]) -> str:
        """Grouped bar chart with CI error bars for all datasets."""
        n = len(all_results)
        x = np.arange(n)
        width = 0.2
        labels = [r.dataset_name.split("(")[0].strip() for r in all_results]

        fig, ax = plt.subplots(figsize=(max(14, n * 1.6), 6))

        for i, metric in enumerate(self.METRICS):
            means = [getattr(r, f"ci_{metric}").mean for r in all_results]
            lowers = [getattr(r, f"ci_{metric}").lower for r in all_results]
            uppers = [getattr(r, f"ci_{metric}").upper for r in all_results]
            errs = [
                [m - lo for m, lo in zip(means, lowers)],
                [hi - m for m, hi in zip(means, uppers)]
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
            "KAN Baseline Performance — All Datasets (5-Fold CV, 95 % CI)",
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

    # ── per-fold line chart per dataset ────────────────────────────────────

    def plot_fold_line(self, result: DatasetResult) -> str:
        """Line chart showing metric trend across folds."""
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
        ax.set_title(
            f"Per-Fold Metrics Trend — {result.dataset_name}",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        safe_name = result.dataset_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        path = os.path.join(self.save_dir, f"{safe_name}_fold_line.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESULT EXPORTER  (JSON + CSV)
# ─────────────────────────────────────────────────────────────────────────────

class ResultExporter:
    """Serializes DatasetResult objects to JSON and CSV."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def export_json(self, all_results: List[DatasetResult]) -> str:
        payload = []
        for r in all_results:
            entry = {
                "dataset": r.dataset_name,
                "n_samples": r.n_samples,
                "n_features": r.n_features,
                "n_classes": r.n_classes,
                "cross_validation": {
                    "folds": [
                        {
                            "fold": fm.fold,
                            "accuracy": fm.accuracy,
                            "precision": fm.precision,
                            "recall": fm.recall,
                            "f1": fm.f1,
                            "train_loss": fm.train_loss,
                            "test_loss": fm.test_loss
                        }
                        for fm in r.fold_metrics
                    ],
                    "confidence_intervals_95": {
                        metric: {
                            "mean": getattr(r, f"ci_{metric}").mean,
                            "lower": getattr(r, f"ci_{metric}").lower,
                            "upper": getattr(r, f"ci_{metric}").upper,
                            "std": getattr(r, f"ci_{metric}").std
                        }
                        for metric in ["accuracy", "precision", "recall", "f1"]
                    }
                },
                "holdout_test": {
                    "accuracy": r.holdout_accuracy,
                    "precision": r.holdout_precision,
                    "recall": r.holdout_recall,
                    "f1": r.holdout_f1
                }
            }
            payload.append(entry)

        path = os.path.join(self.save_dir, "all_results.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        return path

    def export_csv_summary(self, all_results: List[DatasetResult]) -> str:
        rows = []
        for r in all_results:
            row = {
                "Dataset": r.dataset_name,
                "N_Samples": r.n_samples,
                "N_Features": r.n_features,
                "N_Classes": r.n_classes,
            }
            for metric in ["accuracy", "precision", "recall", "f1"]:
                ci = getattr(r, f"ci_{metric}")
                half = (ci.upper - ci.lower) / 2
                row[f"CV_{metric}_mean"] = round(ci.mean, 4)
                row[f"CV_{metric}_CI95_lower"] = round(ci.lower, 4)
                row[f"CV_{metric}_CI95_upper"] = round(ci.upper, 4)
                row[f"CV_{metric}_pm"] = round(half, 4)
                row[f"CV_{metric}_std"] = round(ci.std, 4)
            row["Holdout_Accuracy"] = round(r.holdout_accuracy, 4)
            row["Holdout_Precision"] = round(r.holdout_precision, 4)
            row["Holdout_Recall"] = round(r.holdout_recall, 4)
            row["Holdout_F1"] = round(r.holdout_f1, 4)
            rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.save_dir, "summary_table.csv")
        df.to_csv(path, index=False)
        return path

    def export_csv_per_fold(self, all_results: List[DatasetResult]) -> str:
        rows = []
        for r in all_results:
            for fm in r.fold_metrics:
                rows.append({
                    "Dataset": r.dataset_name,
                    "Fold": fm.fold,
                    "Accuracy": round(fm.accuracy, 4),
                    "Precision": round(fm.precision, 4),
                    "Recall": round(fm.recall, 4),
                    "F1": round(fm.f1, 4),
                    "Train_Loss": round(fm.train_loss, 4),
                    "Test_Loss": round(fm.test_loss, 4)
                })

        df = pd.DataFrame(rows)
        path = os.path.join(self.save_dir, "per_fold_metrics.csv")
        df.to_csv(path, index=False)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PIPELINE ORCHESTRATOR  (ties everything together)
# ─────────────────────────────────────────────────────────────────────────────

class KANBaselinePipeline:
    """
    Main orchestrator.
    Runs load → CV → holdout → visualize → export for every dataset.
    """

    KAN_PARAMS = {
        "n_hidden": 5,
        "grid": 5,
        "k": 4,
        "lr": 0.01,
        "lamb": 0.01,
        "steps": 50
    }

    def __init__(
        self,
        save_dir: str = "./kan_baseline_results",
        n_folds: int = 5,
        holdout_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.save_dir = save_dir
        self.n_folds = n_folds
        self.holdout_size = holdout_size
        self.random_state = random_state
        self.verbose = verbose

        os.makedirs(save_dir, exist_ok=True)

        self.cv = CrossValidator(n_splits=n_folds, random_state=random_state)
        self.viz = ResultVisualizer(save_dir)
        self.exporter = ResultExporter(save_dir)

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

    # ── helpers ────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _separator(self) -> None:
        self._log("─" * 70)

    # ── per-dataset processing ─────────────────────────────────────────────

    def _process_dataset(self, loader: BaseDatasetLoader) -> Optional[DatasetResult]:
        self._separator()
        self._log(f"  Dataset : {loader.name}")

        # Load data
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

        # Hold-out split (stratified)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=self.holdout_size,
            random_state=self.random_state,
            stratify=y
        )

        # ── 5-Fold CV on training portion ────────────────────────────────
        self._log(f"\n  Running {self.n_folds}-Fold CV …")
        fold_metrics = self.cv.run(
            X_train_full, y_train_full,
            num_features=n_features,
            num_classes=n_classes,
            kan_params=self.KAN_PARAMS,
            verbose=self.verbose
        )
        result.fold_metrics = fold_metrics
        result.compute_confidence_intervals()

        # ── Final holdout evaluation ──────────────────────────────────────
        self._log("\n  Training final model on full training set …")
        scaler_final = StandardScaler()
        X_tr_s = scaler_final.fit_transform(X_train_full)
        X_te_s = scaler_final.transform(X_test)

        final_trainer = KANTrainer(
            num_features=n_features,
            num_classes=n_classes,
            params=self.KAN_PARAMS,
            seed=self.random_state
        )
        final_trainer.fit(X_tr_s, y_train_full, X_te_s, y_test)
        test_preds = final_trainer.predict(X_te_s)

        avg = "binary" if n_classes == 2 else "weighted"
        result.holdout_accuracy = accuracy_score(y_test, test_preds)
        result.holdout_precision = precision_score(y_test, test_preds, average=avg, zero_division=0)
        result.holdout_recall = recall_score(y_test, test_preds, average=avg, zero_division=0)
        result.holdout_f1 = f1_score(y_test, test_preds, average=avg, zero_division=0)

        self._log(
            f"\n  [HOLDOUT] Acc={result.holdout_accuracy:.4f} | "
            f"Prec={result.holdout_precision:.4f} | "
            f"Rec={result.holdout_recall:.4f} | "
            f"F1={result.holdout_f1:.4f}"
        )

        # Classification report
        self._log(
            f"\n  Classification Report (Holdout):\n"
            + classification_report(y_test, test_preds, target_names=target_names, zero_division=0)
        )

        # CI summary
        self._log("  95% Confidence Intervals (CV):")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            ci = getattr(result, f"ci_{metric}")
            self._log(f"    {metric.capitalize():12s}: {ci}")

        # ── Visualizations ────────────────────────────────────────────────
        self._log("\n  Generating visualizations …")
        self.viz.plot_fold_boxplot(result)
        self.viz.plot_fold_line(result)
        self.viz.plot_confusion_matrix(result, y_test, test_preds, target_names)

        return result

    # ── main entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        self._log("\n" + "=" * 70)
        self._log("  KAN Baseline Multi-Dataset Evaluator")
        self._log(f"  Device : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self._log(f"  Params : {json.dumps(self.KAN_PARAMS)}")
        self._log("=" * 70)

        for loader in self.loaders:
            result = self._process_dataset(loader)
            if result is not None:
                self.all_results.append(result)

        # ── Summary visualizations ────────────────────────────────────────
        if self.all_results:
            self._separator()
            self._log("\n  Generating summary comparison chart …")
            self.viz.plot_summary_comparison(self.all_results)

            # ── Export ───────────────────────────────────────────────────
            self._log("  Exporting results …")
            p_json = self.exporter.export_json(self.all_results)
            p_csv = self.exporter.export_csv_summary(self.all_results)
            p_fold_csv = self.exporter.export_csv_per_fold(self.all_results)

            self._log(f"    JSON   → {p_json}")
            self._log(f"    CSV    → {p_csv}")
            self._log(f"    Folds  → {p_fold_csv}")

            # ── Print final summary table ────────────────────────────────
            self._print_summary_table()

        self._log("\n  All done!  Results saved to: " + self.save_dir)

    def _print_summary_table(self) -> None:
        self._separator()
        self._log("FINAL SUMMARY  (5-Fold CV — 95% Confidence Intervals)")
        self._separator()
        header = (
            f"{'Dataset':<35} {'Acc (mean ± CI)':<22} "
            f"{'Prec (mean ± CI)':<22} {'Rec (mean ± CI)':<22} "
            f"{'F1 (mean ± CI)':<22}"
        )
        self._log(header)
        self._log("-" * len(header))

        for r in self.all_results:
            cols = []
            for metric in ["accuracy", "precision", "recall", "f1"]:
                ci = getattr(r, f"ci_{metric}")
                half = (ci.upper - ci.lower) / 2
                cols.append(f"{ci.mean:.4f} ± {half:.4f}")
            row = f"{r.dataset_name:<35} {cols[0]:<22} {cols[1]:<22} {cols[2]:<22} {cols[3]:<22}"
            self._log(row)

        self._log("-" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Google Colab Drive integration (optional) ──────────────────────────
    try:
        from google.colab import drive as _drive
        _drive.mount("/content/drive")
        SAVE_DIR = "/content/drive/MyDrive/Colab_Notebooks/medical/kan_baseline_results_v2"
    except (ImportError, Exception):
        # Running locally
        SAVE_DIR = "./kan_baseline_results_v2"

    pipeline = KANBaselinePipeline(
        save_dir=SAVE_DIR,
        n_folds=5,
        holdout_size=0.2,
        random_state=42,
        verbose=True
    )
    pipeline.run()
