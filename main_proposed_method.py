"""
KAN Proposed Method — Optuna HPO + Ensemble Soft Voting
=========================================================
Object-Oriented implementation of the PROPOSED METHOD that combines:

  STRATEGY A — Baseline (Single KAN, seed=42, fixed hyperparameters)
  STRATEGY B — Baseline + Optuna HPO (Single KAN, Optuna-tuned hyperparameters)
  STRATEGY C — Proposed: Optuna HPO + Ensemble Soft Voting
               (5 KAN sub-models, Optuna-tuned hyperparameters, seeds 0-4,
                averaged probability logits before argmax)

All three strategies share the SAME train/test split and CV folds for
a fair apples-to-apples comparison across 9 medical datasets.

Pipeline architecture (leakage-free):
  1. train_test_split (stratified, 20%) → [train_full | holdout_test]
     → holdout_test is SEALED until the final evaluation
  2. Optuna (Strategies B & C): inner 3-fold CV on train_full only
     → scaler fit inside every inner fold (no leakage)
  3. Outer 5-fold CV for all 3 strategies on train_full
     → scaler fit inside every outer fold (no leakage)
  4. Final holdout evaluation
     → scaler fit on entire train_full → evaluate on holdout_test

Evaluation methodology:
  - 5-Fold Stratified Cross-Validation (per-fold metrics stored)
  - 95% Confidence Intervals via t-distribution (correct for small n=5)
  - Hold-out test set (20%, stratified, fixed split)
  - Box plots + line charts per dataset per strategy
  - Side-by-side comparison plots (all 3 strategies)
  - Confusion matrices for all strategies
  - Delta charts showing improvement of proposed method over baselines
  - Global heatmaps across all datasets
  - JSON + CSV export of all results

Datasets (9):
  1. Breast Cancer (sklearn)        6. Dermatology
  2. Hepatitis                       7. WPBC
  3. Liver (BUPA)                    8. Heart Disease (Cleveland)
  4. Parkinson's (Telemonitoring)    9. CTG (Cardiotocography)
  5. Diabetes (Pima)

Usage:
    python kan_proposed_optuna_ensemble.py
"""

import os
import json
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
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldMetrics:
    """Metrics for one CV fold of one strategy."""
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_loss: float
    val_loss: float


@dataclass
class ConfidenceInterval:
    """95% CI for a single metric computed via t-distribution."""
    mean: float
    lower: float
    upper: float
    std: float

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} ± {self.half_width:.4f}"
            f"  [{self.lower:.4f}, {self.upper:.4f}]"
        )


@dataclass
class StrategyResult:
    """
    Full result for ONE strategy on ONE dataset.
    Stores per-fold CV metrics, computed CIs, and final holdout scores.
    """
    strategy_name: str           # "Baseline", "Optuna", or "Optuna+Ensemble"
    dataset_name: str
    n_samples: int
    n_features: int
    n_classes: int
    best_params: Dict = field(default_factory=dict)

    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    ci_accuracy:  Optional[ConfidenceInterval] = None
    ci_precision: Optional[ConfidenceInterval] = None
    ci_recall:    Optional[ConfidenceInterval] = None
    ci_f1:        Optional[ConfidenceInterval] = None

    holdout_accuracy:  float = 0.0
    holdout_precision: float = 0.0
    holdout_recall:    float = 0.0
    holdout_f1:        float = 0.0
    holdout_loss:      float = 0.0

    def compute_confidence_intervals(self, confidence: float = 0.95) -> None:
        """Fill ci_* fields using t-distribution (correct for k=5 folds)."""
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = np.array([getattr(fm, metric) for fm in self.fold_metrics])
            n = len(values)
            mean = values.mean()
            std = values.std(ddof=1)
            t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
            margin = t_crit * (std / np.sqrt(n))
            ci = ConfidenceInterval(
                mean=mean,
                lower=max(0.0, mean - margin),
                upper=min(1.0, mean + margin),
                std=std
            )
            setattr(self, f"ci_{metric}", ci)


@dataclass
class DatasetTriple:
    """Bundles all three strategy results for the same dataset."""
    dataset_name: str
    target_names: List[str]
    baseline: StrategyResult
    optuna_single: StrategyResult
    optuna_ensemble: StrategyResult

    # Holdout ground-truth and predictions for confusion matrices
    y_test: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_preds: np.ndarray = field(default_factory=lambda: np.array([]))
    optuna_preds: np.ndarray = field(default_factory=lambda: np.array([]))
    ensemble_preds: np.ndarray = field(default_factory=lambda: np.array([]))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET LOADERS
# ═════════════════════════════════════════════════════════════════════════════

class BaseDatasetLoader(ABC):
    """Abstract interface for all dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns (X, y_encoded_0based, target_names)."""
        ...

    def _encode_and_clean(
        self,
        df: pd.DataFrame,
        target_col: str,
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shared preprocessing: drop cols → replace '?' → encode → impute."""
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
        le_t = LabelEncoder()
        y = le_t.fit_transform(y_raw)
        return X, y


class BreastCancerLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Breast Cancer"

    def load(self):
        data = load_breast_cancer()
        return data.data, data.target, list(data.target_names)


class HepatitisLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Hepatitis"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/hepatitis/hepatitis.data")
        cols = [
            "Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue",
            "Malaise", "Anorexia", "LiverBig", "LiverFirm", "SpleenPalpable",
            "Spiders", "Ascites", "Varices", "Bilirubin", "AlkPhosphate",
            "SGOT", "Albumin", "Protime", "Histology"
        ]
        df = pd.read_csv(url, names=cols)
        X, y = self._encode_and_clean(df, target_col="Class")
        return X, y, ["Die", "Live"]


class LiverLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Liver (BUPA)"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/liver-disorders/bupa.data")
        cols = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
        df = pd.read_csv(url, names=cols)
        df["selector"] = df["selector"] - 1
        X = df.drop(columns=["selector"]).values.astype(float)
        y = df["selector"].values
        return X, y, ["Group 1", "Group 2"]


class ParkinsonsLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Parkinson's"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/parkinsons/telemonitoring/parkinsons_updrs.data")
        df = pd.read_csv(url)
        median_val = df["total_UPDRS"].median()
        df["target"] = (df["total_UPDRS"] >= median_val).astype(int)
        drop = ["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"]
        X, y = self._encode_and_clean(df, target_col="target", drop_cols=drop)
        return X, y, ["Mild", "Severe"]


class DiabetesLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Diabetes (Pima)"

    def load(self):
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets"
               "/master/pima-indians-diabetes.data.csv")
        cols = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
        ]
        df = pd.read_csv(url, names=cols)
        X = df.drop(columns=["Outcome"]).values.astype(float)
        y = df["Outcome"].values
        return X, y, ["Non-Diabetic", "Diabetic"]


class DermatologyLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "Dermatology"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/dermatology/dermatology.data")
        df = pd.read_csv(url, header=None)
        df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["Class"]
        X, y = self._encode_and_clean(df, target_col="Class")
        return X, y, [f"Class {i+1}" for i in range(len(np.unique(y)))]


class WPBCLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "WPBC"

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
    def name(self) -> str:
        return "Heart Disease"

    def load(self):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/heart-disease/processed.cleveland.data")
        cols = [
            "Age", "Sex", "CP", "Trestbps", "Chol", "Fbs", "Restecg",
            "Thalach", "Exang", "Oldpeak", "Slope", "Ca", "Thal", "Target"
        ]
        df = pd.read_csv(url, names=cols)
        df["Target"] = df["Target"].apply(
            lambda v: 0 if str(v).strip() == "0" else 1
        )
        X, y = self._encode_and_clean(df, target_col="Target")
        return X, y, ["No Disease", "Disease"]


class CTGLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "CTG"

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

        le_t = LabelEncoder()
        y = le_t.fit_transform(y_raw.values.ravel())
        n_cls = len(np.unique(y))
        return X, y, [f"Class {i+1}" for i in range(n_cls)]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KAN TRAINERS
# ═════════════════════════════════════════════════════════════════════════════

# Default fixed hyperparameters for Strategy A (Baseline)
BASELINE_PARAMS: Dict = {
    "n_hidden": 5,
    "grid":     5,
    "k":        4,
    "lr":       0.01,
    "lamb":     0.01,
    "steps":    50,
}


class KANTrainerBase:
    """
    Shared base for all KAN trainers.
    Handles model construction, tensor conversion, and loss computation.
    """

    def __init__(self, num_features: int, num_classes: int, params: Dict):
        self.num_features = num_features
        self.num_classes  = num_classes
        self.params       = params
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn      = nn.CrossEntropyLoss()

    def _build_model(self, seed: int) -> KAN:
        return KAN(
            width=[self.num_features, self.params["n_hidden"], self.num_classes],
            grid=self.params["grid"],
            k=self.params["k"],
            seed=seed,
            device=self.device
        )

    def _to_tensor(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(X).float().to(self.device),
            torch.from_numpy(y).long().to(self.device)
        )

    def _x_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(X).float().to(self.device)

    def _train_one(
        self,
        model: KAN,
        X_tr_t: torch.Tensor,
        y_tr_t: torch.Tensor,
        X_va_t: torch.Tensor,
        y_va_t: torch.Tensor
    ) -> Tuple[float, float]:
        """Fit one KAN model; return (train_loss, val_loss)."""
        model.fit(
            {
                "train_input": X_tr_t,
                "train_label": y_tr_t,
                "test_input":  X_va_t,
                "test_label":  y_va_t,
            },
            opt="LBFGS",
            steps=self.params["steps"],
            lr=self.params["lr"],
            loss_fn=self.loss_fn,
            lamb=self.params["lamb"]
        )
        with torch.no_grad():
            tr_loss = self.loss_fn(model(X_tr_t), y_tr_t).item()
            va_loss = self.loss_fn(model(X_va_t), y_va_t).item()
        return tr_loss, va_loss


class SingleKANTrainer(KANTrainerBase):
    """
    Single KAN model trainer (used by Strategy A & B).
    Strategy A uses fixed BASELINE_PARAMS with seed=42.
    Strategy B uses Optuna-tuned params.
    """

    def __init__(self, num_features: int, num_classes: int,
                 params: Dict, seed: int = 42):
        super().__init__(num_features, num_classes, params)
        self.seed  = seed
        self.model: Optional[KAN] = None

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray
    ) -> Tuple[float, float]:
        self.model = self._build_model(self.seed)
        X_tr_t, y_tr_t = self._to_tensor(X_train, y_train)
        X_va_t, y_va_t = self._to_tensor(X_val,   y_val)
        return self._train_one(self.model, X_tr_t, y_tr_t, X_va_t, y_va_t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = self._x_tensor(X)
        with torch.no_grad():
            return torch.argmax(self.model(X_t), dim=1).cpu().numpy()

    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X_t, y_t = self._to_tensor(X, y)
        with torch.no_grad():
            return self.loss_fn(self.model(X_t), y_t).item()


class EnsembleKANTrainer(KANTrainerBase):
    """
    Strategy C — Soft Voting Ensemble with Optuna-tuned params.

    Trains N sub-models (seeds 0…N-1) using the Optuna best params.
    Prediction = argmax( mean of all sub-models' softmax logit vectors ).
    This is 'soft voting': we average class probability distributions,
    not hard labels, which is more robust than majority voting.
    """

    def __init__(
        self,
        num_features: int,
        num_classes:  int,
        params:       Dict,
        n_members:    int = 5
    ):
        super().__init__(num_features, num_classes, params)
        self.n_members = n_members
        self.models:   List[KAN] = []

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray
    ) -> Tuple[float, float]:
        """Train all sub-models; return ensemble (train_loss, val_loss)."""
        self.models = []
        X_tr_t, y_tr_t = self._to_tensor(X_train, y_train)
        X_va_t, y_va_t = self._to_tensor(X_val,   y_val)

        for seed in range(self.n_members):
            model = self._build_model(seed)
            self._train_one(model, X_tr_t, y_tr_t, X_va_t, y_va_t)
            self.models.append(model)

        # Ensemble losses from averaged logits
        with torch.no_grad():
            avg_tr = torch.stack([m(X_tr_t) for m in self.models]).mean(dim=0)
            avg_va = torch.stack([m(X_va_t) for m in self.models]).mean(dim=0)
            tr_loss = self.loss_fn(avg_tr, y_tr_t).item()
            va_loss = self.loss_fn(avg_va, y_va_t).item()

        return tr_loss, va_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Soft voting: average logits across all sub-models, then argmax."""
        X_t = self._x_tensor(X)
        with torch.no_grad():
            avg_logits = torch.stack(
                [m(X_t) for m in self.models]
            ).mean(dim=0)
            return torch.argmax(avg_logits, dim=1).cpu().numpy()

    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X_t, y_t = self._to_tensor(X, y)
        with torch.no_grad():
            avg_logits = torch.stack(
                [m(X_t) for m in self.models]
            ).mean(dim=0)
            return self.loss_fn(avg_logits, y_t).item()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTUNA HPO (leakage-free inner loop)
# ═════════════════════════════════════════════════════════════════════════════

class OptunaOptimizer:
    """
    Runs Optuna TPE search on an INNER cross-validation of the training set.

    Leakage-prevention guarantees:
    • Receives only X_train_full / y_train_full — holdout_test is NEVER seen.
    • Each trial creates its own inner StratifiedKFold, fits a fresh scaler
      per inner fold on inner-train only, and evaluates on inner-val only.
    • No scaler, model, or statistic from any trial leaks into the outer CV
      or holdout evaluation.

    Used by BOTH Strategy B (single KAN) and Strategy C (ensemble) since they
    share the same hyperparameter space and objective (maximize inner-CV F1).
    The best params are then fixed for the outer CV and final holdout.
    """

    def __init__(
        self,
        n_trials:      int = 50,
        n_inner_folds: int = 3,
        random_state:  int = 42
    ):
        self.n_trials      = n_trials
        self.n_inner_folds = n_inner_folds
        self.random_state  = random_state

    def _make_objective(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        num_features: int,
        num_classes:  int
    ):
        """Closure that returns the Optuna objective function."""
        avg_mode = "binary" if num_classes == 2 else "weighted"
        inner_skf = StratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=self.random_state
        )

        def objective(trial: optuna.Trial) -> float:
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

                # Scaler fit ONLY on inner-train
                scaler = StandardScaler()
                X_i_tr_s = scaler.fit_transform(X_i_tr)
                X_i_va_s = scaler.transform(X_i_va)

                trainer = SingleKANTrainer(
                    num_features=num_features,
                    num_classes=num_classes,
                    params=params,
                    seed=self.random_state + fold_i
                )
                try:
                    trainer.fit(X_i_tr_s, y_i_tr, X_i_va_s, y_i_va)
                    preds = trainer.predict(X_i_va_s)
                    fold_f1s.append(
                        f1_score(y_i_va, preds, average=avg_mode, zero_division=0)
                    )
                except Exception:
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_f1s))

        return objective

    def run(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        num_features: int,
        num_classes:  int,
        verbose: bool = True
    ) -> Tuple[Dict, optuna.Study]:
        """Run Optuna; return (best_params, study)."""
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study   = optuna.create_study(direction="maximize", sampler=sampler)
        objective = self._make_objective(
            X_train_full, y_train_full, num_features, num_classes
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        if verbose:
            print(
                f"    [Optuna] Best inner-CV F1 = {study.best_value:.4f}  |  "
                f"Params: {best}"
            )
        return best, study


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CROSS-VALIDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class CrossValidator:
    """
    Runs Stratified K-Fold CV for a given trainer factory function.
    Scaler is fit inside each fold — no leakage.
    Returns a list of FoldMetrics, one per fold.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits     = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trainer_factory,        # callable() → SingleKANTrainer | EnsembleKANTrainer
        num_classes: int,
        strategy_label: str = "",
        verbose: bool = True
    ) -> List[FoldMetrics]:
        avg_mode = "binary" if num_classes == 2 else "weighted"
        fold_metrics: List[FoldMetrics] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(
            self.skf.split(X, y), start=1
        ):
            if verbose:
                print(
                    f"    [{strategy_label}] Fold {fold_idx}/{self.n_splits} ...",
                    end=" ", flush=True
                )

            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            scaler   = StandardScaler()
            X_tr_s   = scaler.fit_transform(X_tr)
            X_va_s   = scaler.transform(X_va)

            trainer  = trainer_factory()
            tr_loss, va_loss = trainer.fit(X_tr_s, y_tr, X_va_s, y_va)
            preds    = trainer.predict(X_va_s)

            fm = FoldMetrics(
                fold=fold_idx,
                accuracy=accuracy_score(y_va, preds),
                precision=precision_score(y_va, preds, average=avg_mode, zero_division=0),
                recall=recall_score(y_va, preds, average=avg_mode, zero_division=0),
                f1=f1_score(y_va, preds, average=avg_mode, zero_division=0),
                train_loss=tr_loss,
                val_loss=va_loss
            )
            fold_metrics.append(fm)

            if verbose:
                print(
                    f"Acc={fm.accuracy:.4f} | Prec={fm.precision:.4f} | "
                    f"Rec={fm.recall:.4f} | F1={fm.f1:.4f}"
                )

        return fold_metrics


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALIZER
# ═════════════════════════════════════════════════════════════════════════════

class ResultVisualizer:
    """Generates and saves all plots."""

    METRICS = ["accuracy", "precision", "recall", "f1"]

    METRIC_COLORS = {
        "accuracy":  "#2563EB",
        "precision": "#16A34A",
        "recall":    "#D97706",
        "f1":        "#9333EA",
    }

    # Colors per strategy
    STRATEGY_COLORS = {
        "Baseline":       "#E53E3E",   # red
        "Optuna":         "#D97706",   # amber
        "Optuna+Ensemble": "#2B6CB0",  # blue  ← proposed method
    }

    STRATEGY_LABELS = {
        "Baseline":       "Baseline (Fixed HP)",
        "Optuna":         "Optuna HPO (Single KAN)",
        "Optuna+Ensemble": "Optuna + Ensemble (Proposed)",
    }

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _safe(name: str) -> str:
        return (name.replace(" ", "_").replace("/", "-")
                    .replace("(", "").replace(")", "")
                    .replace("+", "plus"))

    # ── 1. Per-fold box plot (one strategy, one dataset) ──────────────────────

    def plot_fold_boxplot(self, result: StrategyResult) -> str:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
        strategy_label = self.STRATEGY_LABELS.get(result.strategy_name, result.strategy_name)
        fig.suptitle(
            f"Per-Fold Metrics [{strategy_label}]\n"
            f"{result.dataset_name}  |  5-Fold CV  |  n={result.n_samples}",
            fontsize=11, fontweight="bold", y=1.01
        )

        for ax, metric in zip(axes, self.METRICS):
            values = [getattr(fm, metric) for fm in result.fold_metrics]
            ci     = getattr(result, f"ci_{metric}")
            color  = self.METRIC_COLORS[metric]

            bp = ax.boxplot(
                values, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=color, linewidth=1.5),
                capprops=dict(color=color, linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor=color, markersize=5)
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.72)

            rng = np.random.default_rng(seed=0)
            jitter = rng.uniform(-0.07, 0.07, len(values))
            for j, v in enumerate(values):
                ax.scatter(1 + jitter[j], v, color=color, zorder=5,
                           s=38, edgecolors="white", linewidths=0.7)
                ax.annotate(f"F{j+1}", (1 + jitter[j], v),
                            textcoords="offset points", xytext=(5, 0),
                            fontsize=7, color="#555")

            ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
            ax.set_xticks([])
            lo = max(0.0, min(values) - 0.08)
            hi = min(1.0, max(values) + 0.08)
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.2f}")
            )
            ax.set_xlabel(
                f"Mean={ci.mean:.4f}\n95% CI [{ci.lower:.4f}, {ci.upper:.4f}]",
                fontsize=8
            )
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fname = (f"{self._safe(result.dataset_name)}"
                 f"_{self._safe(result.strategy_name)}_boxplot.png")
        path  = os.path.join(self.save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 2. Per-fold line chart (one strategy, one dataset) ───────────────────

    def plot_fold_line(self, result: StrategyResult) -> str:
        folds  = [fm.fold for fm in result.fold_metrics]
        strategy_label = self.STRATEGY_LABELS.get(result.strategy_name, result.strategy_name)
        fig, ax = plt.subplots(figsize=(8, 4))

        for metric in self.METRICS:
            values = [getattr(fm, metric) for fm in result.fold_metrics]
            ax.plot(folds, values, marker="o", label=metric.capitalize(),
                    color=self.METRIC_COLORS[metric], linewidth=1.8, markersize=6)

        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            f"Per-Fold Trend [{strategy_label}]\n{result.dataset_name}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fname = (f"{self._safe(result.dataset_name)}"
                 f"_{self._safe(result.strategy_name)}_line.png")
        path  = os.path.join(self.save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 3. Confusion matrix ───────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        result:       StrategyResult,
        y_true:       np.ndarray,
        y_pred:       np.ndarray,
        target_names: List[str],
        cmap:         str = "Blues"
    ) -> str:
        cm  = confusion_matrix(y_true, y_pred)
        n   = len(target_names)
        strategy_label = self.STRATEGY_LABELS.get(result.strategy_name, result.strategy_name)
        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.3)))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title(
            f"Confusion Matrix [{strategy_label}]\n"
            f"{result.dataset_name}  |  Holdout Acc={result.holdout_accuracy:.4f}",
            fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()

        fname = (f"{self._safe(result.dataset_name)}"
                 f"_{self._safe(result.strategy_name)}_cm.png")
        path  = os.path.join(self.save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 4. Three-way comparison bar chart (per dataset) ───────────────────────

    def plot_three_way_comparison(self, triple: DatasetTriple) -> str:
        """
        Grouped bar chart comparing all 3 strategies on all 4 metrics,
        with 95% CI error bars.  The proposed method (Optuna+Ensemble) is
        highlighted in blue.
        """
        x       = np.arange(len(self.METRICS))
        width   = 0.25
        labels  = [m.capitalize() for m in self.METRICS]
        strategies = [
            ("Baseline",        triple.baseline,        "Baseline (Fixed HP)"),
            ("Optuna",          triple.optuna_single,   "Optuna HPO (Single)"),
            ("Optuna+Ensemble", triple.optuna_ensemble, "Optuna+Ensemble [Proposed]"),
        ]

        fig, ax = plt.subplots(figsize=(11, 6))

        for i, (key, result, disp_label) in enumerate(strategies):
            means  = [getattr(result, f"ci_{m}").mean  for m in self.METRICS]
            lowers = [getattr(result, f"ci_{m}").lower for m in self.METRICS]
            uppers = [getattr(result, f"ci_{m}").upper for m in self.METRICS]
            errs   = [
                [m - lo for m, lo in zip(means, lowers)],
                [hi - m  for m, hi in zip(means, uppers)]
            ]
            color = self.STRATEGY_COLORS[key]
            edgecolor = "#1a1a1a" if key == "Optuna+Ensemble" else "none"
            lw        = 1.5      if key == "Optuna+Ensemble" else 0
            bars = ax.bar(
                x + i * width, means, width,
                label=disp_label,
                color=color, alpha=0.85,
                edgecolor=edgecolor, linewidth=lw,
                yerr=errs, capsize=5,
                error_kw={"elinewidth": 1.4, "ecolor": "#222"}
            )
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.013,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333"
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score (95% CI)", fontsize=11)
        ax.set_title(
            f"Three-Way Comparison — {triple.dataset_name}\n"
            f"(5-Fold CV, 95% CI error bars)",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fname = f"{self._safe(triple.dataset_name)}_three_way_comparison.png"
        path  = os.path.join(self.save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 5. Optuna optimisation history ────────────────────────────────────────

    def plot_optuna_history(
        self,
        dataset_name: str,
        best_params:  Dict,
        study:        optuna.Study
    ) -> str:
        trials = [
            t for t in study.trials
            if t.value is not None
            and t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not trials:
            return ""

        nums        = [t.number for t in trials]
        values      = [t.value  for t in trials]
        best_so_far = [max(values[:i+1]) for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(nums, values, alpha=0.5, s=20, color="#6B7280", label="Trial F1")
        ax.plot(nums, best_so_far, color="#DC2626", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial number", fontsize=10)
        ax.set_ylabel("Inner-CV F1 (mean)", fontsize=10)
        ax.set_title(
            f"Optuna History — {dataset_name}\n"
            f"Best F1={max(values):.4f}  |  Params: {best_params}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        fname = f"{self._safe(dataset_name)}_optuna_history.png"
        path  = os.path.join(self.save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 6. Global heatmap (all datasets × all metrics) ────────────────────────

    def plot_global_heatmap(self, all_triples: List[DatasetTriple]) -> str:
        """
        Three side-by-side heatmaps: Baseline | Optuna | Optuna+Ensemble.
        """
        dataset_labels = [t.dataset_name for t in all_triples]
        col_labels     = [m.capitalize() for m in self.METRICS]

        def _matrix(key: str) -> np.ndarray:
            rows = []
            for triple in all_triples:
                result = getattr(triple, key)
                rows.append([
                    getattr(result, f"ci_{m}").mean for m in self.METRICS
                ])
            return np.array(rows)

        matrices = [
            (_matrix("baseline"),        "Baseline (Fixed HP)",         "Reds"),
            (_matrix("optuna_single"),   "Optuna HPO (Single KAN)",     "Oranges"),
            (_matrix("optuna_ensemble"), "Optuna+Ensemble [Proposed]",  "Blues"),
        ]

        fig, axes = plt.subplots(
            1, 3,
            figsize=(20, max(5, len(all_triples) * 0.65))
        )
        for ax, (mat, title, cmap) in zip(axes, matrices):
            sns.heatmap(
                mat, ax=ax,
                xticklabels=col_labels,
                yticklabels=dataset_labels,
                annot=True, fmt=".3f", cmap=cmap,
                vmin=0.5, vmax=1.0,
                linewidths=0.4, linecolor="#ccc",
                cbar_kws={"shrink": 0.7}
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Metric", fontsize=10)
            ax.tick_params(axis="y", labelsize=9)

        fig.suptitle(
            "CV Mean Score Heatmap — All Datasets × All Strategies (5-Fold, 95% CI means)",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        path = os.path.join(self.save_dir, "global_heatmap_three_way.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 7. Global grouped bar (all datasets, all strategies) ──────────────────

    def plot_global_summary_bar(self, all_triples: List[DatasetTriple]) -> str:
        n        = len(all_triples)
        x        = np.arange(n)
        width    = 0.25
        d_labels = [t.dataset_name.split("(")[0].strip() for t in all_triples]

        fig, axes = plt.subplots(2, 2, figsize=(max(18, n * 2.1), 10))
        axes = axes.flatten()

        strat_map = [
            ("Baseline",        "baseline",        "Baseline (Fixed HP)"),
            ("Optuna",          "optuna_single",   "Optuna (Single KAN)"),
            ("Optuna+Ensemble", "optuna_ensemble", "Optuna+Ensemble [Proposed]"),
        ]

        for ax, metric in zip(axes, self.METRICS):
            for i, (key, attr, disp_label) in enumerate(strat_map):
                means  = [getattr(getattr(t, attr), f"ci_{metric}").mean  for t in all_triples]
                lowers = [getattr(getattr(t, attr), f"ci_{metric}").lower for t in all_triples]
                uppers = [getattr(getattr(t, attr), f"ci_{metric}").upper for t in all_triples]
                errs   = [
                    [m - lo for m, lo in zip(means, lowers)],
                    [hi - m  for m, hi in zip(means, uppers)]
                ]
                ax.bar(
                    x + i * width, means, width,
                    label=disp_label,
                    color=self.STRATEGY_COLORS[key],
                    alpha=0.82,
                    yerr=errs, capsize=4,
                    error_kw={"elinewidth": 1.2, "ecolor": "#333"}
                )

            ax.set_xticks(x + width)
            ax.set_xticklabels(d_labels, rotation=28, ha="right", fontsize=8)
            ax.set_ylim(0, 1.18)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            "Three-Way Comparison — All Datasets (5-Fold CV, 95% CI)\n"
            "Baseline  vs  Optuna HPO  vs  Optuna+Ensemble [Proposed]",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        path = os.path.join(self.save_dir, "global_summary_bar_three_way.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 8. Delta chart (Proposed vs each baseline) ───────────────────────────

    def plot_delta_charts(self, all_triples: List[DatasetTriple]) -> List[str]:
        """
        Two delta charts:
          (a) Proposed − Baseline
          (b) Proposed − Optuna-Single
        """
        paths     = []
        n         = len(all_triples)
        d_labels  = [t.dataset_name.split("(")[0].strip() for t in all_triples]
        comp_map  = [
            ("vs_baseline", "baseline",      "Proposed − Baseline",      "#16A34A"),
            ("vs_optuna",   "optuna_single", "Proposed − Optuna (Single)", "#2563EB"),
        ]

        for suffix, attr, title_prefix, default_pos_color in comp_map:
            fig, axes = plt.subplots(
                1, 4,
                figsize=(20, max(4, n * 0.55)),
                sharey=True
            )
            for ax, metric in zip(axes, self.METRICS):
                deltas = [
                    getattr(t.optuna_ensemble, f"ci_{metric}").mean
                    - getattr(getattr(t, attr), f"ci_{metric}").mean
                    for t in all_triples
                ]
                colors = ["#16A34A" if d >= 0 else "#DC2626" for d in deltas]
                y_pos  = np.arange(n)
                ax.barh(y_pos, deltas, color=colors, alpha=0.82)
                ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
                ax.set_yticks(y_pos)
                if ax == axes[0]:
                    ax.set_yticklabels(d_labels, fontsize=9)
                ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
                ax.set_xlabel("Δ Score", fontsize=8)
                ax.grid(axis="x", alpha=0.3, linestyle="--")
                ax.spines[["top", "right"]].set_visible(False)

                for i, d in enumerate(deltas):
                    ax.text(
                        d + (0.002 if d >= 0 else -0.002),
                        i, f"{d:+.3f}",
                        va="center",
                        ha="left" if d >= 0 else "right",
                        fontsize=7
                    )

            fig.suptitle(
                f"{title_prefix}  (green=improvement, red=regression)",
                fontsize=12, fontweight="bold"
            )
            plt.tight_layout()
            path = os.path.join(self.save_dir, f"delta_{suffix}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

        return paths


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RESULT EXPORTER
# ═════════════════════════════════════════════════════════════════════════════

class ResultExporter:
    """Serialises all results to JSON and CSV."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _strategy_dict(r: StrategyResult) -> dict:
        folds = [
            {
                "fold":       fm.fold,
                "accuracy":   fm.accuracy,
                "precision":  fm.precision,
                "recall":     fm.recall,
                "f1":         fm.f1,
                "train_loss": fm.train_loss,
                "val_loss":   fm.val_loss,
            }
            for fm in r.fold_metrics
        ]
        cis = {
            m: {
                "mean":  getattr(r, f"ci_{m}").mean,
                "lower": getattr(r, f"ci_{m}").lower,
                "upper": getattr(r, f"ci_{m}").upper,
                "std":   getattr(r, f"ci_{m}").std,
            }
            for m in ["accuracy", "precision", "recall", "f1"]
        }
        return {
            "best_params":                r.best_params,
            "cv_folds":                   folds,
            "cv_confidence_intervals_95": cis,
            "holdout": {
                "accuracy":  r.holdout_accuracy,
                "precision": r.holdout_precision,
                "recall":    r.holdout_recall,
                "f1":        r.holdout_f1,
                "loss":      r.holdout_loss,
            }
        }

    def export_json(self, all_triples: List[DatasetTriple]) -> str:
        payload = []
        for t in all_triples:
            payload.append({
                "dataset":        t.dataset_name,
                "n_samples":      t.baseline.n_samples,
                "n_features":     t.baseline.n_features,
                "n_classes":      t.baseline.n_classes,
                "baseline":       self._strategy_dict(t.baseline),
                "optuna_single":  self._strategy_dict(t.optuna_single),
                "optuna_ensemble": self._strategy_dict(t.optuna_ensemble),
            })
        path = os.path.join(self.save_dir, "all_results_three_way.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        return path

    def export_csv_summary(self, all_triples: List[DatasetTriple]) -> str:
        rows = []
        for t in all_triples:
            row: Dict = {
                "Dataset":    t.dataset_name,
                "N_Samples":  t.baseline.n_samples,
                "N_Features": t.baseline.n_features,
                "N_Classes":  t.baseline.n_classes,
            }
            for disp, key in [
                ("Base",    "baseline"),
                ("Optuna",  "optuna_single"),
                ("PropEns", "optuna_ensemble"),
            ]:
                r = getattr(t, key)
                for m in ["accuracy", "precision", "recall", "f1"]:
                    ci = getattr(r, f"ci_{m}")
                    row[f"{disp}_{m}_mean"]  = round(ci.mean,       4)
                    row[f"{disp}_{m}_pm"]    = round(ci.half_width, 4)
                    row[f"{disp}_{m}_lower"] = round(ci.lower,      4)
                    row[f"{disp}_{m}_upper"] = round(ci.upper,      4)
                    row[f"{disp}_{m}_std"]   = round(ci.std,        4)
                row[f"{disp}_holdout_acc"]   = round(r.holdout_accuracy,  4)
                row[f"{disp}_holdout_prec"]  = round(r.holdout_precision, 4)
                row[f"{disp}_holdout_rec"]   = round(r.holdout_recall,    4)
                row[f"{disp}_holdout_f1"]    = round(r.holdout_f1,        4)
            rows.append(row)
        path = os.path.join(self.save_dir, "summary_table_three_way.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_csv_per_fold(self, all_triples: List[DatasetTriple]) -> str:
        rows = []
        for t in all_triples:
            for strategy, key in [
                ("Baseline",        "baseline"),
                ("Optuna",          "optuna_single"),
                ("Optuna+Ensemble", "optuna_ensemble"),
            ]:
                r = getattr(t, key)
                for fm in r.fold_metrics:
                    rows.append({
                        "Dataset":    t.dataset_name,
                        "Strategy":   strategy,
                        "Fold":       fm.fold,
                        "Accuracy":   round(fm.accuracy,   4),
                        "Precision":  round(fm.precision,  4),
                        "Recall":     round(fm.recall,     4),
                        "F1":         round(fm.f1,         4),
                        "Train_Loss": round(fm.train_loss, 4),
                        "Val_Loss":   round(fm.val_loss,   4),
                    })
        path = os.path.join(self.save_dir, "per_fold_metrics_three_way.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_csv_delta(self, all_triples: List[DatasetTriple]) -> str:
        """
        Delta table: Proposed − Baseline and Proposed − Optuna per metric.
        Positive = improvement of the proposed method.
        """
        rows = []
        for t in all_triples:
            row: Dict = {"Dataset": t.dataset_name}
            for m in ["accuracy", "precision", "recall", "f1"]:
                base_mean   = getattr(t.baseline,        f"ci_{m}").mean
                optuna_mean = getattr(t.optuna_single,   f"ci_{m}").mean
                prop_mean   = getattr(t.optuna_ensemble, f"ci_{m}").mean
                row[f"delta_vs_baseline_{m}"] = round(prop_mean - base_mean,   4)
                row[f"delta_vs_optuna_{m}"]   = round(prop_mean - optuna_mean, 4)
            rows.append(row)
        path = os.path.join(self.save_dir, "delta_table_three_way.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class KANProposedPipeline:
    """
    Main orchestrator — three-strategy, leakage-free pipeline per dataset.

    ┌──────────────────────────────────────────────────────────────┐
    │ STAGE 0 — Data loading & hold-out split (ONCE per dataset)   │
    │   train_test_split (stratified, 20%)                         │
    │   → X_train_full / y_train_full   (used for HPO + CV)        │
    │   → X_test / y_test               (SEALED until stage 3)     │
    ├──────────────────────────────────────────────────────────────┤
    │ STRATEGY A — Baseline (fixed hyperparameters)                │
    │   Stage 1A: 5-fold outer CV on X_train_full                  │
    │   Stage 2A: Final model → evaluate on X_test                 │
    ├──────────────────────────────────────────────────────────────┤
    │ STRATEGY B — Optuna HPO + Single KAN                         │
    │   Stage 1B: Optuna (50 trials, 3-fold inner CV) on           │
    │             X_train_full → best_params                        │
    │   Stage 2B: 5-fold outer CV with best_params                 │
    │   Stage 3B: Final model → evaluate on X_test                 │
    ├──────────────────────────────────────────────────────────────┤
    │ STRATEGY C — Proposed: Optuna HPO + Ensemble Soft Voting     │
    │   Reuses best_params from Strategy B (same Optuna run)       │
    │   Stage 2C: 5-fold outer CV with best_params + 5-model ens.  │
    │   Stage 3C: Final ensemble → evaluate on X_test              │
    └──────────────────────────────────────────────────────────────┘

    Notes:
    - All three strategies use THE SAME train/test split (same random_state).
    - The same StratifiedKFold split is reused across strategies per dataset.
    - Strategies B and C share ONE Optuna run (same best_params).
    - The only difference between B and C is single vs. ensemble inference.
    """

    N_ENSEMBLE_MEMBERS = 5

    def __init__(
        self,
        save_dir:      str   = "./kan_proposed_results",
        n_folds:       int   = 5,
        holdout_size:  float = 0.2,
        n_trials:      int   = 50,
        n_inner_folds: int   = 3,
        random_state:  int   = 42,
        verbose:       bool  = True
    ):
        self.save_dir      = save_dir
        self.n_folds       = n_folds
        self.holdout_size  = holdout_size
        self.n_trials      = n_trials
        self.random_state  = random_state
        self.verbose       = verbose

        os.makedirs(save_dir, exist_ok=True)

        self.optuna_opt = OptunaOptimizer(
            n_trials=n_trials,
            n_inner_folds=n_inner_folds,
            random_state=random_state
        )
        self.cv  = CrossValidator(n_splits=n_folds, random_state=random_state)
        self.viz = ResultVisualizer(save_dir)
        self.exp = ResultExporter(save_dir)

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

        self.all_triples: List[DatasetTriple] = []

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _sep(self, char: str = "─", n: int = 78) -> None:
        self._log(char * n)

    # ── trainer factories ─────────────────────────────────────────────────────

    def _baseline_factory(self, num_features: int, num_classes: int):
        def factory():
            return SingleKANTrainer(
                num_features=num_features,
                num_classes=num_classes,
                params=BASELINE_PARAMS.copy(),
                seed=self.random_state
            )
        return factory

    def _optuna_single_factory(self, num_features: int, num_classes: int, best_params: Dict):
        def factory():
            return SingleKANTrainer(
                num_features=num_features,
                num_classes=num_classes,
                params=best_params.copy(),
                seed=self.random_state
            )
        return factory

    def _optuna_ensemble_factory(self, num_features: int, num_classes: int, best_params: Dict):
        n = self.N_ENSEMBLE_MEMBERS
        def factory():
            return EnsembleKANTrainer(
                num_features=num_features,
                num_classes=num_classes,
                params=best_params.copy(),
                n_members=n
            )
        return factory

    # ── per-dataset processing ────────────────────────────────────────────────

    def _process_dataset(self, loader: BaseDatasetLoader) -> Optional[DatasetTriple]:
        self._sep()
        self._log(f"  Dataset : {loader.name}")

        try:
            X, y, target_names = loader.load()
        except Exception as exc:
            self._log(f"  [ERROR] Could not load {loader.name}: {exc}")
            return None

        n_samples, n_features = X.shape
        n_classes  = len(np.unique(y))
        avg_mode   = "binary" if n_classes == 2 else "weighted"

        self._log(f"  Samples={n_samples}  Features={n_features}  Classes={n_classes}")

        # ── STAGE 0: Hold-out carve-out ──────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.holdout_size,
            random_state=self.random_state,
            stratify=y
        )

        # ── Prepare StrategyResult containers ────────────────────────────────
        def make_result(name: str) -> StrategyResult:
            return StrategyResult(
                strategy_name=name,
                dataset_name=loader.name,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes
            )

        base_result = make_result("Baseline")
        opt_result  = make_result("Optuna")
        ens_result  = make_result("Optuna+Ensemble")

        # ════════════════════════════════════════════════════════════════════
        # STRATEGY A — Baseline (fixed hyperparameters, no HPO)
        # ════════════════════════════════════════════════════════════════════
        self._log(f"\n  [Strategy A — Baseline] Running {self.n_folds}-Fold CV …")
        base_result.best_params   = BASELINE_PARAMS.copy()
        base_result.fold_metrics  = self.cv.run(
            X_train, y_train,
            trainer_factory=self._baseline_factory(n_features, n_classes),
            num_classes=n_classes,
            strategy_label="Baseline",
            verbose=self.verbose
        )
        base_result.compute_confidence_intervals()

        # ════════════════════════════════════════════════════════════════════
        # OPTUNA HPO — shared by Strategies B & C
        # ════════════════════════════════════════════════════════════════════
        self._log(
            f"\n  [Optuna HPO] {self.n_trials} trials, "
            f"{self.optuna_opt.n_inner_folds}-fold inner CV …"
        )
        best_params, study = self.optuna_opt.run(
            X_train, y_train,
            num_features=n_features,
            num_classes=n_classes,
            verbose=self.verbose
        )
        opt_result.best_params = best_params
        ens_result.best_params = best_params

        # ════════════════════════════════════════════════════════════════════
        # STRATEGY B — Optuna HPO + Single KAN
        # ════════════════════════════════════════════════════════════════════
        self._log(f"\n  [Strategy B — Optuna Single] Running {self.n_folds}-Fold CV …")
        opt_result.fold_metrics = self.cv.run(
            X_train, y_train,
            trainer_factory=self._optuna_single_factory(n_features, n_classes, best_params),
            num_classes=n_classes,
            strategy_label="Optuna",
            verbose=self.verbose
        )
        opt_result.compute_confidence_intervals()

        # ════════════════════════════════════════════════════════════════════
        # STRATEGY C — Proposed: Optuna HPO + Ensemble Soft Voting
        # ════════════════════════════════════════════════════════════════════
        self._log(
            f"\n  [Strategy C — Proposed: Optuna+Ensemble] "
            f"Running {self.n_folds}-Fold CV "
            f"({self.N_ENSEMBLE_MEMBERS} sub-models per fold) …"
        )
        ens_result.fold_metrics = self.cv.run(
            X_train, y_train,
            trainer_factory=self._optuna_ensemble_factory(n_features, n_classes, best_params),
            num_classes=n_classes,
            strategy_label="Optuna+Ensemble",
            verbose=self.verbose
        )
        ens_result.compute_confidence_intervals()

        # ════════════════════════════════════════════════════════════════════
        # FINAL HOLDOUT EVALUATION — all three strategies
        # Scaler fit on all of X_train (no leakage); X_test never seen before.
        # ════════════════════════════════════════════════════════════════════
        self._log("\n  Final holdout evaluation …")
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_train)
        X_te_s   = scaler.transform(X_test)

        # Strategy A holdout
        self._log("    [Strategy A — Baseline] …")
        base_trainer = SingleKANTrainer(
            n_features, n_classes, BASELINE_PARAMS.copy(), seed=self.random_state
        )
        base_trainer.fit(X_tr_s, y_train, X_te_s, y_test)
        base_preds = base_trainer.predict(X_te_s)
        base_result.holdout_accuracy  = accuracy_score(y_test, base_preds)
        base_result.holdout_precision = precision_score(y_test, base_preds, average=avg_mode, zero_division=0)
        base_result.holdout_recall    = recall_score(y_test, base_preds, average=avg_mode, zero_division=0)
        base_result.holdout_f1        = f1_score(y_test, base_preds, average=avg_mode, zero_division=0)
        base_result.holdout_loss      = base_trainer.predict_loss(X_te_s, y_test)

        # Strategy B holdout
        self._log("    [Strategy B — Optuna Single] …")
        opt_trainer = SingleKANTrainer(
            n_features, n_classes, best_params.copy(), seed=self.random_state
        )
        opt_trainer.fit(X_tr_s, y_train, X_te_s, y_test)
        opt_preds = opt_trainer.predict(X_te_s)
        opt_result.holdout_accuracy  = accuracy_score(y_test, opt_preds)
        opt_result.holdout_precision = precision_score(y_test, opt_preds, average=avg_mode, zero_division=0)
        opt_result.holdout_recall    = recall_score(y_test, opt_preds, average=avg_mode, zero_division=0)
        opt_result.holdout_f1        = f1_score(y_test, opt_preds, average=avg_mode, zero_division=0)
        opt_result.holdout_loss      = opt_trainer.predict_loss(X_te_s, y_test)

        # Strategy C holdout (proposed)
        self._log("    [Strategy C — Optuna+Ensemble (Proposed)] …")
        ens_trainer = EnsembleKANTrainer(
            n_features, n_classes, best_params.copy(),
            n_members=self.N_ENSEMBLE_MEMBERS
        )
        ens_trainer.fit(X_tr_s, y_train, X_te_s, y_test)
        ens_preds = ens_trainer.predict(X_te_s)
        ens_result.holdout_accuracy  = accuracy_score(y_test, ens_preds)
        ens_result.holdout_precision = precision_score(y_test, ens_preds, average=avg_mode, zero_division=0)
        ens_result.holdout_recall    = recall_score(y_test, ens_preds, average=avg_mode, zero_division=0)
        ens_result.holdout_f1        = f1_score(y_test, ens_preds, average=avg_mode, zero_division=0)
        ens_result.holdout_loss      = ens_trainer.predict_loss(X_te_s, y_test)

        # ── Holdout summary ───────────────────────────────────────────────────
        self._log("\n  Holdout results:")
        for label, result in [
            ("Baseline",             base_result),
            ("Optuna (Single)",       opt_result),
            ("Optuna+Ensemble [Prop]", ens_result),
        ]:
            self._log(
                f"    [{label}] "
                f"Acc={result.holdout_accuracy:.4f} | "
                f"Prec={result.holdout_precision:.4f} | "
                f"Rec={result.holdout_recall:.4f} | "
                f"F1={result.holdout_f1:.4f}"
            )

        # ── CI summary ────────────────────────────────────────────────────────
        self._log("\n  95% CIs (outer CV):")
        for label, result in [
            ("Baseline",                base_result),
            ("Optuna (Single)",          opt_result),
            ("Optuna+Ensemble [Prop]",   ens_result),
        ]:
            self._log(f"    [{label}]")
            for m in ["accuracy", "precision", "recall", "f1"]:
                self._log(f"      {m.capitalize():<12}: {getattr(result, f'ci_{m}')}")

        # ── Classification reports ────────────────────────────────────────────
        for label, result, preds in [
            ("Baseline",                base_result, base_preds),
            ("Optuna (Single)",          opt_result,  opt_preds),
            ("Optuna+Ensemble [Prop]",   ens_result,  ens_preds),
        ]:
            self._log(f"\n  [{label}] Classification Report (Holdout):")
            self._log(classification_report(
                y_test, preds,
                target_names=target_names,
                zero_division=0
            ))

        # ── Build triple ──────────────────────────────────────────────────────
        triple = DatasetTriple(
            dataset_name=loader.name,
            target_names=target_names,
            baseline=base_result,
            optuna_single=opt_result,
            optuna_ensemble=ens_result,
            y_test=y_test,
            baseline_preds=base_preds,
            optuna_preds=opt_preds,
            ensemble_preds=ens_preds
        )

        # ── Per-dataset visualizations ────────────────────────────────────────
        self._log("\n  Generating per-dataset visualizations …")
        for result in [base_result, opt_result, ens_result]:
            self.viz.plot_fold_boxplot(result)
            self.viz.plot_fold_line(result)

        cmaps = {"Baseline": "Greens", "Optuna": "Oranges", "Optuna+Ensemble": "Blues"}
        for label, result, preds in [
            ("Baseline",         base_result, base_preds),
            ("Optuna",           opt_result,  opt_preds),
            ("Optuna+Ensemble",  ens_result,  ens_preds),
        ]:
            self.viz.plot_confusion_matrix(
                result, y_test, preds, target_names,
                cmap=cmaps[label]
            )

        self.viz.plot_three_way_comparison(triple)
        self.viz.plot_optuna_history(loader.name, best_params, study)

        return triple

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        self._sep("═")
        self._log("  KAN Proposed Method — Optuna HPO + Ensemble Soft Voting")
        self._log("  Three-way comparison: Baseline | Optuna (Single) | Proposed")
        self._log(f"  Device         : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self._log(f"  Baseline Params: {json.dumps(BASELINE_PARAMS)}")
        self._log(f"  Optuna Trials  : {self.n_trials}  (inner {self.optuna_opt.n_inner_folds}-fold CV)")
        self._log(f"  Ensemble Size  : {self.N_ENSEMBLE_MEMBERS} sub-models (seeds 0–{self.N_ENSEMBLE_MEMBERS-1})")
        self._log(f"  Outer CV Folds : {self.n_folds}")
        self._log(f"  Hold-out       : {int(self.holdout_size*100)}%")
        self._sep("═")

        for loader in self.loaders:
            triple = self._process_dataset(loader)
            if triple is not None:
                self.all_triples.append(triple)

        # ── Global visualizations ─────────────────────────────────────────────
        if self.all_triples:
            self._sep()
            self._log("\n  Generating global visualizations …")
            self.viz.plot_global_summary_bar(self.all_triples)
            self.viz.plot_global_heatmap(self.all_triples)
            self.viz.plot_delta_charts(self.all_triples)

            # ── Export ───────────────────────────────────────────────────────
            self._log("  Exporting results …")
            p1 = self.exp.export_json(self.all_triples)
            p2 = self.exp.export_csv_summary(self.all_triples)
            p3 = self.exp.export_csv_per_fold(self.all_triples)
            p4 = self.exp.export_csv_delta(self.all_triples)
            self._log(f"    JSON             → {p1}")
            self._log(f"    Summary CSV      → {p2}")
            self._log(f"    Per-fold CSV     → {p3}")
            self._log(f"    Delta CSV        → {p4}")

            # ── Console summary tables ────────────────────────────────────────
            self._print_summary_table("Baseline",                "baseline")
            self._print_summary_table("Optuna HPO (Single KAN)", "optuna_single")
            self._print_summary_table("Proposed: Optuna+Ensemble", "optuna_ensemble")
            self._print_delta_table()

        self._log(f"\n  All done!  Results saved to: {self.save_dir}\n")

    # ── console table helpers ─────────────────────────────────────────────────

    def _print_summary_table(self, label: str, key: str) -> None:
        self._sep()
        self._log(f"SUMMARY — {label}  (5-Fold CV, 95% CI)")
        self._sep()
        hdr = (f"{'Dataset':<28} {'Accuracy':^22} {'Precision':^22}"
               f" {'Recall':^22} {'F1':^22}")
        self._log(hdr)
        self._log("-" * len(hdr))
        for t in self.all_triples:
            r    = getattr(t, key)
            cols = []
            for m in ["accuracy", "precision", "recall", "f1"]:
                ci = getattr(r, f"ci_{m}")
                cols.append(f"{ci.mean:.4f} ± {ci.half_width:.4f}")
            self._log(
                f"{t.dataset_name:<28} {cols[0]:^22} {cols[1]:^22}"
                f" {cols[2]:^22} {cols[3]:^22}"
            )
        self._log("-" * len(hdr))

    def _print_delta_table(self) -> None:
        self._sep()
        self._log("DELTA TABLE — Proposed (Optuna+Ensemble) vs Baseline & Optuna-Single")
        self._sep()
        self._log(f"  {'Dataset':<28}  {'Metric':<12}  {'vs Baseline':>13}  {'vs Optuna':>13}")
        self._log("-" * 74)
        for t in self.all_triples:
            for m in ["accuracy", "precision", "recall", "f1"]:
                b  = getattr(t.baseline,        f"ci_{m}").mean
                o  = getattr(t.optuna_single,   f"ci_{m}").mean
                p  = getattr(t.optuna_ensemble, f"ci_{m}").mean
                vs_b = f"{p - b:+.4f}"
                vs_o = f"{p - o:+.4f}"
                ds   = t.dataset_name if m == "accuracy" else ""
                self._log(f"  {ds:<28}  {m.capitalize():<12}  {vs_b:>13}  {vs_o:>13}")
            self._log("")
        self._log("-" * 74)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Google Colab auto-detection
    try:
        from google.colab import drive as _drive
        _drive.mount("/content/drive")
        SAVE_DIR = (
            "/content/drive/MyDrive/Colab_Notebooks/medical"
            "/kan_proposed_optuna_ensemble"
        )
    except (ImportError, Exception):
        SAVE_DIR = "./kan_proposed_optuna_ensemble"

    pipeline = KANProposedPipeline(
        save_dir      = SAVE_DIR,
        n_folds       = 5,       # outer CV folds
        holdout_size  = 0.2,     # 20% sealed holdout test set
        n_trials      = 50,      # Optuna trials per dataset
        n_inner_folds = 3,       # inner CV folds inside Optuna
        random_state  = 42,
        verbose       = True
    )
    pipeline.run()
