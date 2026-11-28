"""Подбор параметров распределений и сравнение их по AIC/BIC."""
from dataclasses import dataclass

import numpy as np
from scipy import stats


# Стартовый набор распределений.
DEFAULT_DISTRIBUTIONS = [
    "norm",         # Гаусс
    "rayleigh",     # Рэлей
    "expon",        # экспоненциальное
    "gamma",        # гамма
    "lognorm",      # логнормальное
    "weibull_min",  # Вейбулла
    "arcsine",      # Арксинусное (для синусоид)
    "nakagami",     # Накагами (замирания)
    "rice",         # Райса (сигнал+шум)
    "skewnorm",     # Скошенное нормальное
    "gennorm",      # Обобщённое нормальное
    "laplace",      # Лапласа
    "t",            # Стьюдента
]



@dataclass
class FitResult:
    name: str                       
    params: tuple = ()              
    param_names: tuple = ()         
    aic: float = np.inf
    bic: float = np.inf
    loglik: float = -np.inf
    weight_percent: float = 0.0     
    error: str | None = None        


def _param_names(dist) -> tuple[str, ...]:
    """Имена параметров: shape-параметры + loc + scale."""
    shapes = getattr(dist, "shapes", None)
    shape_names = [s.strip() for s in shapes.split(",")] if shapes else []
    return tuple(shape_names + ["loc", "scale"])


import warnings  

def _fit_one(dist_name: str, data: np.ndarray) -> FitResult:
    dist = getattr(stats, dist_name)
    names = _param_names(dist)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = dist.fit(data)
    except Exception as e:                      
        return FitResult(name=dist_name, error=f"fit failed: {e}")

    if not np.all(np.isfinite(params)):
        return FitResult(name=dist_name, params=tuple(params),
                         param_names=names, error="non-finite params")

    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = dist.pdf(data, *params)
        pdf = np.clip(pdf, 1e-300, None)       
        loglik = float(np.sum(np.log(pdf)))

    if not np.isfinite(loglik):
        return FitResult(name=dist_name, params=tuple(params),
                         param_names=names, error="non-finite loglik")

    k = len(params)                             
    n = len(data)                             
    aic = 2.0 * k - 2.0 * loglik
    bic = k * np.log(n) - 2.0 * loglik

    return FitResult(
        name=dist_name,
        params=tuple(float(p) for p in params),
        param_names=names,
        aic=float(aic),
        bic=float(bic),
        loglik=loglik,
    )


def compute_akaike_weights(aic_values) -> np.ndarray:
    """Веса Акаике в процентах: чем меньше AIC, тем больше вес."""
    aic = np.asarray(aic_values, dtype=float)
    finite = np.isfinite(aic)
    if not finite.any():
        return np.zeros_like(aic)

    delta = np.where(finite, aic - aic[finite].min(), np.inf)
    w = np.where(finite, np.exp(-0.5 * delta), 0.0)
    total = w.sum()
    if total == 0:
        return np.zeros_like(aic)
    return 100.0 * w / total


def fit_all(data, distributions: list[str] | None = None) -> list[FitResult]:
    """Подогнать все распределения и вернуть список, отсортированный по AIC."""
    data = np.asarray(data, dtype=float)
    if data.size < 10:
        raise ValueError("Слишком маленькая выборка (< 10 значений)")

    names = distributions or DEFAULT_DISTRIBUTIONS
    results = [_fit_one(name, data) for name in names]

    aics = np.array([r.aic for r in results])
    weights = compute_akaike_weights(aics)
    for r, w in zip(results, weights):
        r.weight_percent = float(w)

    results.sort(key=lambda r: (r.aic, r.name))
    return results


def format_table(results: list[FitResult]) -> str:
    """Текстовая таблица для CLI и отладки."""
    lines = [f"{'distribution':<14} {'AIC':>12} {'BIC':>12} {'weight %':>10}  params"]
    for r in results:
        if r.error:
            lines.append(f"{r.name:<14} {'—':>12} {'—':>12} {'—':>10}  ({r.error})")
        else:
            pstr = ", ".join(f"{v:.4g}" for v in r.params)
            lines.append(
                f"{r.name:<14} {r.aic:>12.4f} {r.bic:>12.4f} "
                f"{r.weight_percent:>10.2f}  ({pstr})"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from src.data_loader import load_data

    if len(sys.argv) < 2:
        print("Использование: python -m src.distribution_fitter <файл> [столбец]")
        sys.exit(1)
    file_arg = sys.argv[1]
    col_arg = sys.argv[2] if len(sys.argv) > 2 else None

    arr = load_data(file_arg, col_arg)
    print(f"Данные: {len(arr)} значений")
    results = fit_all(arr)
    print(format_table(results))