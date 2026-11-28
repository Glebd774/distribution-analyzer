"""Сгенерировать тестовый CSV: нормальное, рэлеевское, экспоненциальное."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
n = 1000

df = pd.DataFrame({
    "normal":      rng.normal(loc=10, scale=2, size=n),
    "rayleigh":    rng.rayleigh(scale=3, size=n),
    "exponential": rng.exponential(scale=2, size=n),
})
df.to_csv("data/sample_data.csv", index=False)
print("Готово: data/sample_data.csv")