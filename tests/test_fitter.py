"""Юнит-тесты для distribution_fitter."""
import numpy as np
import pytest
from scipy import stats

from src.distribution_fitter import (
    DEFAULT_DISTRIBUTIONS,
    compute_akaike_weights,
    fit_all,
)


def _gen(dist_name: str, n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist_name == "norm": return rng.normal(loc=5, scale=2, size=n)
    if dist_name == "rayleigh": return rng.rayleigh(scale=3, size=n)
    if dist_name == "expon": return rng.exponential(scale=2, size=n)
    if dist_name == "gamma": return rng.gamma(shape=2.0, scale=1.5, size=n)
    if dist_name == "lognorm": return rng.lognormal(mean=0, sigma=0.5, size=n)
    if dist_name == "weibull_min": return rng.weibull(a=2.0, size=n)
    if dist_name == "laplace": return rng.laplace(size=n)
    if dist_name == "t": return rng.standard_t(df=5, size=n)
    if dist_name == "arcsine": return stats.arcsine.rvs(size=n, random_state=seed)
    if dist_name == "nakagami": return stats.nakagami.rvs(nu=2.0, size=n, random_state=seed)
    if dist_name == "rice": return stats.rice.rvs(b=2.0, scale=1.0, size=n, random_state=seed)
    if dist_name == "gennorm": return stats.gennorm.rvs(beta=1.5, size=n, random_state=seed) # beta=1.5, чтобы не совпало с Гауссом
    if dist_name == "skewnorm": return stats.skewnorm.rvs(a=5.0, size=n, random_state=seed)
    
    raise ValueError(dist_name)


def test_weights_sum_to_100():
    w = compute_akaike_weights([100.0, 101.0, 105.0])
    assert w.sum() == pytest.approx(100.0)


def test_best_model_gets_largest_weight():
    w = compute_akaike_weights([100.0, 110.0, 120.0])
    assert w[0] > w[1] > w[2]
    assert w[0] > 99.0


def test_weights_with_inf_are_zero():
    w = compute_akaike_weights([100.0, np.inf, 101.0])
    assert w[1] == 0.0
    assert w.sum() == pytest.approx(100.0)


@pytest.mark.parametrize("true_dist", ["norm", "rayleigh", "expon", "arcsine", "nakagami", "rice", "gennorm", "laplace", "t"])
def test_correct_distribution_wins(true_dist):
    """Для данных из известного распределения оно должно быть в топ-2."""
    data = _gen(true_dist)
    results = fit_all(data)
    top_two = [r.name for r in results[:2]]
    assert true_dist in top_two, f"{true_dist} не попал в топ-2: {top_two}"


def test_fit_all_sorted_by_aic():
    data = _gen("norm")
    results = fit_all(data)
    aics = [r.aic for r in results]
    assert aics == sorted(aics)


def test_small_sample_raises():
    with pytest.raises(ValueError):
        fit_all([1.0, 2.0, 3.0])


def test_custom_distribution_list():
    data = _gen("norm")
    results = fit_all(data, distributions=["norm", "expon"])
    names = {r.name for r in results}
    assert names == {"norm", "expon"}


def test_result_fields_are_finite():
    data = _gen("norm")
    results = fit_all(data)
    for r in results:
        if r.error is None:
            assert np.isfinite(r.aic)
            assert np.isfinite(r.bic)
            assert np.isfinite(r.loglik)
            assert len(r.params) == len(r.param_names)

def test_norm_params_close_to_truth():
    data = _gen("norm", n=5000, seed=42)
    results = fit_all(data, distributions=["norm"])
    r = results[0]
    # params = (loc, scale) = (5, 2)
    assert r.params[0] == pytest.approx(5.0, abs=0.2)
    assert r.params[1] == pytest.approx(2.0, abs=0.2)


def test_default_distributions_contains_required():
    assert "norm" in DEFAULT_DISTRIBUTIONS
    assert "rayleigh" in DEFAULT_DISTRIBUTIONS