"""Загрузка числовых данных из файлов разных форматов."""
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls", ".json"}


def _read_raw(file_path: str | Path) -> pd.DataFrame:
    """Прочитать файл в DataFrame, выбрав подходящий парсер по расширению."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат: {ext}. "
            f"Поддерживаются: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".txt":
        # sep=None + engine='python' — автоопределение разделителя
        return pd.read_csv(path, sep=None, engine="python")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)

    raise ValueError(f"Внутренняя ошибка: не обработан формат {ext}")


def list_numeric_columns(file_path: str | Path) -> list[str]:
    """Вернуть имена числовых столбцов в файле (для выпадающего списка в GUI)."""
    df = _read_raw(file_path)
    # Берем только числовые столбцы, в которых есть хотя бы одно непустое значение
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [col for col in numeric_cols if not df[col].isna().all()]
    return valid_cols


def load_data(file_path: str | Path, column_name: str | None = None) -> np.ndarray:
    """
    Загрузить числовые данные из файла.

    Parameters
    ----------
    file_path : str | Path
        Путь к файлу (csv/txt/xlsx/xls/json).
    column_name : str | None
        Имя столбца. Если None — берётся первый числовой столбец.

    Returns
    -------
    np.ndarray
        Одномерный массив float, без NaN.
    """
    df = _read_raw(file_path)

    if column_name is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("В файле нет числовых столбцов")
        column_name = numeric_cols[0]
    elif column_name not in df.columns:
        raise ValueError(
            f"Столбец '{column_name}' не найден. "
            f"Доступные: {list(df.columns)}"
        )

    series = df[column_name]
    series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        raise ValueError(f"Столбец '{column_name}' не содержит числовых значений")

    return series.to_numpy(dtype=float)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python -m src.data_loader <файл> [столбец]")
        sys.exit(1)

    file_arg = sys.argv[1]
    col_arg = sys.argv[2] if len(sys.argv) > 2 else None

    print("Числовые столбцы:", list_numeric_columns(file_arg))
    arr = load_data(file_arg, col_arg)
    print(f"Загружено {len(arr)} значений")
    print(f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")