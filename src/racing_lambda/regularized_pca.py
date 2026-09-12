"""Regularized PCA core for the racing leading-signal market layer.

The 0.1:0.9 ratio is implemented as the same correlation regularization
structure used by the research module:

    C_reg = 0.10 * C_recent + 0.90 * C_prior

It is deliberately not treated as a score-mixing ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RECENT_CORRELATION_WEIGHT = 0.10
PRIOR_CORRELATION_WEIGHT = 0.90


def _validate_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{name} must be non-empty")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if frame.shape[0] < 3:
        raise ValueError(f"{name} needs at least three rows")
    if frame.shape[1] < 2:
        raise ValueError(f"{name} needs at least two features")
    return frame.astype(float)


def _standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0)
    if (std <= 1e-12).any():
        bad = list(std.index[std <= 1e-12])
        raise ValueError(f"zero-variance features are not allowed: {bad}")
    return (frame - mean) / std, mean, std


def correlation_matrix(frame: pd.DataFrame) -> np.ndarray:
    standardized, _, _ = _standardize(frame)
    values = standardized.to_numpy(dtype=float)
    corr = values.T @ values / len(values)
    return (corr + corr.T) / 2.0


def regularize_correlation(
    recent_correlation: np.ndarray,
    prior_correlation: np.ndarray,
    recent_weight: float = RECENT_CORRELATION_WEIGHT,
    prior_weight: float = PRIOR_CORRELATION_WEIGHT,
) -> np.ndarray:
    """Blend recent and prior correlations using the fixed 0.1:0.9 ratio."""
    if recent_correlation.shape != prior_correlation.shape:
        raise ValueError("recent and prior correlation matrices must have the same shape")
    if recent_correlation.ndim != 2 or recent_correlation.shape[0] != recent_correlation.shape[1]:
        raise ValueError("correlation matrices must be square")
    if recent_weight < 0 or prior_weight < 0:
        raise ValueError("regularization weights cannot be negative")
    if not np.isclose(recent_weight + prior_weight, 1.0):
        raise ValueError("regularization weights must sum to 1.0")
    combined = recent_weight * recent_correlation + prior_weight * prior_correlation
    return (combined + combined.T) / 2.0


@dataclass(frozen=True)
class PcaProjection:
    scores: pd.DataFrame
    reconstruction_error: pd.Series
    explained_variance_ratio: pd.Series


class RacingRegularizedPCA:
    """PCA over a 0.1 recent / 0.9 prior regularized correlation matrix.

    ``prior_features`` should be a longer historical sample built from the
    same market features as ``recent_features``.  The number of components is
    selected by a cumulative eigenvalue ratio target, which is independent of
    the 0.1:0.9 regularization ratio.
    """

    def __init__(self, variance_target: float = 0.90) -> None:
        if not 0.0 < variance_target <= 1.0:
            raise ValueError("variance_target must be in (0, 1]")
        self.variance_target = float(variance_target)

    def fit(
        self,
        recent_features: pd.DataFrame,
        prior_features: pd.DataFrame,
    ) -> "RacingRegularizedPCA":
        recent = _validate_frame(recent_features, "recent_features")
        prior = _validate_frame(prior_features, "prior_features")
        if list(recent.columns) != list(prior.columns):
            raise ValueError("recent_features and prior_features must use identical columns")

        recent_z, mean, std = _standardize(recent)
        self.columns_ = tuple(recent.columns)
        self.mean_ = mean
        self.std_ = std
        self.recent_correlation_ = correlation_matrix(recent)
        self.prior_correlation_ = correlation_matrix(prior)
        self.regularized_correlation_ = regularize_correlation(
            self.recent_correlation_, self.prior_correlation_
        )

        eigenvalues, eigenvectors = np.linalg.eigh(self.regularized_correlation_)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        eigenvectors = eigenvectors[:, order]
        total = float(eigenvalues.sum())
        if total <= 1e-12:
            raise ValueError("regularized correlation has no positive variance")
        ratios = eigenvalues / total
        cumulative = np.cumsum(ratios)
        n_components = int(np.searchsorted(cumulative, self.variance_target, side="left") + 1)

        self.eigenvalues_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = ratios[:n_components]
        self.components_ = eigenvectors[:, :n_components]
        self.n_components_ = n_components

        recent_values = recent_z.to_numpy(dtype=float)
        recent_scores = recent_values @ self.components_
        self.score_mean_ = recent_scores.mean(axis=0)
        self.score_std_ = recent_scores.std(axis=0, ddof=0)
        self.score_std_ = np.where(self.score_std_ <= 1e-12, 1.0, self.score_std_)
        return self

    def transform(self, features: pd.DataFrame) -> PcaProjection:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before transform")
        frame = _validate_frame(features, "features")
        if list(frame.columns) != list(self.columns_):
            raise ValueError(f"features must use columns in this order: {list(self.columns_)}")

        z = (frame - self.mean_) / self.std_
        values = z.to_numpy(dtype=float)
        score_values = values @ self.components_
        reconstructed = score_values @ self.components_.T
        residual = values - reconstructed
        error = np.sqrt(np.mean(residual**2, axis=1))

        score_columns = [f"PC{i}" for i in range(1, self.n_components_ + 1)]
        return PcaProjection(
            scores=pd.DataFrame(score_values, index=frame.index, columns=score_columns),
            reconstruction_error=pd.Series(error, index=frame.index, name="reconstruction_error"),
            explained_variance_ratio=pd.Series(
                self.explained_variance_ratio_, index=score_columns, name="explained_variance_ratio"
            ),
        )

    def anomaly_score(self, features: pd.DataFrame) -> pd.Series:
        """Return a unit-interval anomaly score from PCA score distance + residual."""
        projection = self.transform(features)
        score_z = (
            projection.scores.to_numpy(dtype=float) - self.score_mean_
        ) / self.score_std_
        distance = np.sqrt(np.mean(score_z**2, axis=1))
        raw = distance + projection.reconstruction_error.to_numpy(dtype=float)
        unit = 1.0 - np.exp(-np.maximum(raw, 0.0))
        return pd.Series(np.clip(unit, 0.0, 1.0), index=features.index, name="market_anomaly")
