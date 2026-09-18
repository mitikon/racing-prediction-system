import numpy as np
import pandas as pd

from racing_lambda.regularized_pca import RacingRegularizedPCA


def _frames(seed: int, n_features: int = 5, n_recent: int = 50, n_prior: int = 200):
    rng = np.random.default_rng(seed)
    columns = [f"f{i}" for i in range(n_features)]
    recent = pd.DataFrame(rng.normal(size=(n_recent, n_features)), columns=columns)
    prior = pd.DataFrame(rng.normal(size=(n_prior, n_features)), columns=columns)
    return recent, prior


def test_n_components_never_exceeds_available_components():
    # variance_target=1.0 forces np.searchsorted to walk past the last
    # cumulative ratio whenever floating-point rounding leaves it just
    # short of 1.0, which previously produced an unclamped n_components_
    # larger than the actual eigenvector count.
    recent, prior = _frames(seed=12)
    pca = RacingRegularizedPCA(variance_target=1.0).fit(recent, prior)
    assert pca.n_components_ <= pca.components_.shape[1]
    assert pca.n_components_ == pca.components_.shape[1]


def test_transform_does_not_raise_a_shape_mismatch_at_the_boundary():
    recent, prior = _frames(seed=12)
    pca = RacingRegularizedPCA(variance_target=1.0).fit(recent, prior)
    projection = pca.transform(recent)
    assert projection.scores.shape[1] == pca.n_components_
    assert len(projection.explained_variance_ratio) == pca.n_components_
