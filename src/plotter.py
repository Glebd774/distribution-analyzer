"""Визуализация подгонки: гистограмма + теоретическая PDF, Q-Q plot."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.distribution_fitter import FitResult


def _x_range_for(data: np.ndarray, pad: float = 0.05) -> np.ndarray:
    """Размах по x: по 0.1-му и 99.9-му процентилям, с запасом."""
    lo, hi = np.percentile(data, [0.1, 99.9])
    span = hi - lo if hi > lo else (abs(lo) or 1.0)
    return np.linspace(lo - pad * span, hi + pad * span, 400)


def plot_fit(data, result: FitResult, bins="auto", figsize=(8, 5)):
    """
    Гистограмма данных + теоретическая PDF выбранного распределения.

    Parameters
    ----------
    data : array-like
    result : FitResult
        Результат из fit_all(). Берётся result.name и result.params.
    bins : int | str
        Число бинов гистограммы (или 'auto').

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)

    # ВОТ ЗДЕСЬ мы используем переменную bins, которую передает GUI
    ax.hist(data, bins=bins, density=True, alpha=0.55,
            color="steelblue", edgecolor="black", label="данные")

    if result.error is None:
        dist = getattr(stats, result.name)
        x = _x_range_for(data)
        try:
            pdf = dist.pdf(x, *result.params)
            ax.plot(x, pdf, "r-", lw=2,
                    label=f"{result.name}  (weight {result.weight_percent:.2f} %)")
        except Exception as exc:                  # noqa: BLE001
            ax.text(0.5, 0.9, f"не удалось построить PDF: {exc}",
                    transform=ax.transAxes, ha="center", color="red")

    ax.set_xlabel("x")
    ax.set_ylabel("плотность вероятности")
    ax.set_title(f"Подгонка: {result.name}   AIC = {result.aic:.2f}")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_qq(data, result: FitResult, figsize=(5, 5)):
    """Q-Q plot: эмпирические квантили против теоретических."""
    data = np.sort(np.asarray(data, dtype=float))
    n = data.size
    probs = (np.arange(1, n + 1) - 0.5) / n

    fig, ax = plt.subplots(figsize=figsize)
    if result.error is not None:
        ax.text(0.5, 0.5, "нет подгонки", ha="center", va="center")
        return fig

    dist = getattr(stats, result.name)
    theoretical = dist.ppf(probs, *result.params)

    ax.scatter(theoretical, data, s=6, alpha=0.5)
    lo = float(min(theoretical.min(), data.min()))
    hi = float(max(theoretical.max(), data.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel("теоретические квантили")
    ax.set_ylabel("эмпирические квантили")
    ax.set_title(f"Q-Q: {result.name}")
    ax.legend()
    fig.tight_layout()
    return fig


def save_fig(fig, path):
    """Сохранить график в файл (png/pdf/svg — по расширению)."""
    path = Path(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


if __name__ == "__main__":
    import sys
    from src.data_loader import load_data
    from src.distribution_fitter import fit_all

    if len(sys.argv) < 2:
        print("Использование: python -m src.plotter <файл> [столбец]")
        sys.exit(1)
    file_arg = sys.argv[1]
    col_arg = sys.argv[2] if len(sys.argv) > 2 else None

    arr = load_data(file_arg, col_arg)
    results = fit_all(arr)
    best = results[0]
    print(f"Лучшая модель: {best.name}  weight = {best.weight_percent:.2f} %")

    tag = col_arg or "first"
    fig1 = plot_fit(arr, best)
    p1 = save_fig(fig1, Path("data") / f"fit_{tag}.png")
    print(f"Сохранено: {p1}")

    fig2 = plot_qq(arr, best)
    p2 = save_fig(fig2, Path("data") / f"qq_{tag}.png")
    print(f"Сохранено: {p2}")