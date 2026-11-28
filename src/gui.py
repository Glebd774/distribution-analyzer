"""Tkinter GUI для анализа распределений (с фильтрацией по номерам строк)."""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")  # важно: до import pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from src.data_loader import list_numeric_columns, load_data
from src.distribution_fitter import FitResult, fit_all
from src.plotter import plot_fit, plot_qq, save_fig


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Анализ распределений физических данных")
        self.geometry("1200x750")  # Размер окна подогнан под новые поля
        self.minsize(1000, 560)

        self.file_path: str | None = None
        self.raw_data = None      # Исходные данные (до обрезки)
        self.data = None          # Обрезанные данные
        self.results: list[FitResult] = []
        self.fit_by_name: dict[str, FitResult] = {}
        self.fig = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Открыть файл…", command=self.open_file).pack(side=tk.LEFT)

        ttk.Label(top, text="Столбец:").pack(side=tk.LEFT, padx=(12, 4))
        self.col_var = tk.StringVar()
        self.col_combo = ttk.Combobox(
            top, textvariable=self.col_var, state="readonly", width=18
        )
        self.col_combo.pack(side=tk.LEFT)
        self.col_combo.bind("<<ComboboxSelected>>", lambda _e: self.analyze())

        ttk.Button(top, text="Анализировать", command=self.analyze).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # --- БЛОК ОБРЕЗКИ ПО НОМЕРАМ СТРОК ---
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(top, text="Строки с:").pack(side=tk.LEFT, padx=(2, 2))
        self.start_idx = tk.StringVar()
        ttk.Entry(top, textvariable=self.start_idx, width=8).pack(side=tk.LEFT)
        
        ttk.Label(top, text="по:").pack(side=tk.LEFT, padx=2)
        self.end_idx = tk.StringVar()
        ttk.Entry(top, textvariable=self.end_idx, width=8).pack(side=tk.LEFT)

        ttk.Label(top, text="Бинов:").pack(side=tk.LEFT, padx=(10, 2))
        self.bins_var = tk.StringVar(value="auto")
        ttk.Entry(top, textvariable=self.bins_var, width=6).pack(side=tk.LEFT)

        ttk.Button(top, text="Сброс", command=self.reset_filters, width=6).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        ttk.Button(top, text="Сброс", command=self.reset_filters, width=6).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        # --------------------------------------

        self.file_label = ttk.Label(top, text="(файл не выбран)", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=12)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(
            self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        ).pack(side=tk.BOTTOM, fill=tk.X)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main, padding=4)
        right = ttk.Frame(main, padding=4)
        main.add(left, weight=1)
        main.add(right, weight=2)

        ttk.Label(left, text="Результаты подгонки (по возрастанию AIC):").pack(
            anchor=tk.W, pady=(0, 4)
        )

        cols = ("name", "aic", "bic", "weight")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=12)
        for col_id, header, w, anchor in (
            ("name", "распределение", 120, tk.W),
            ("aic", "AIC", 90, tk.E),
            ("bic", "BIC", 90, tk.E),
            ("weight", "вес, %", 80, tk.E),
        ):
            self.tree.heading(col_id, text=header)
            self.tree.column(col_id, width=w, anchor=anchor)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        ctrl = ttk.Frame(left, padding=(0, 8, 0, 0))
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Построить:").pack(side=tk.LEFT)
        self.plot_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(
            ctrl, textvariable=self.plot_var, state="readonly", width=18
        )
        self.plot_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="PDF", command=lambda: self.render("pdf")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(ctrl, text="Q-Q", command=lambda: self.render("qq")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(ctrl, text="Сохранить…", command=self.save_current).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self.canvas_frame = ttk.Frame(right)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- actions
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[
                ("Все поддерживаемые", "*.csv *.txt *.xlsx *.xls *.json"),
                ("CSV", "*.csv"),
                ("Текст", "*.txt"),
                ("Excel", "*.xlsx *.xls"),
                ("JSON", "*.json"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        try:
            columns = list_numeric_columns(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{exc}")
            return

        if not columns:
            messagebox.showerror("Ошибка", "В файле нет числовых столбцов")
            return

        self.file_path = path
        self.file_label.config(text=Path(path).name, foreground="black")
        self.col_combo["values"] = columns
        self.col_combo.current(0)
        self.status_var.set(f"Файл загружен: {Path(path).name}")
        
        # Очищаем фильтры при загрузке нового файла
        self.start_idx.set("")
        self.end_idx.set("")
        
        self.analyze()

    def analyze(self):
        if not self.file_path:
            messagebox.showinfo("Нет файла", "Сначала откройте файл с данными")
            return
        column = self.col_var.get()
        if not column:
            return

        try:
            # Загружаем исходные данные и сохраняем их копию
            self.raw_data = load_data(self.file_path, column)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные:\n{exc}")
            return

        self.data = self.raw_data.copy()

        # --- ПРИМЕНЯЕМ ОБРЕЗКУ ПО НОМЕРАМ СТРОК ---
        try:
            start_s = self.start_idx.get().strip()
            end_s = self.end_idx.get().strip()
            
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(self.data)
            
            if start >= 0 and end > start:
                self.data = self.data[start:end]
            else:
                messagebox.showwarning("Внимание", "Некорректный диапазон (начало >= конца). Обрезка пропущена.")
        except ValueError:
            pass # Если ввели текст вместо числа
        # ------------------------------------------

        if len(self.data) < 10:
            messagebox.showwarning(
                "Мало данных", 
                f"После обрезки осталось {len(self.data)} точек. Слишком мало для анализа!"
            )
            return

        self.status_var.set(f"Подгонка распределений для {len(self.data)} точек...")
        self.update_idletasks()

        try:
            self.results = fit_all(self.data)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Подгонка не удалась:\n{exc}")
            self.status_var.set("Ошибка")
            return

        self.fit_by_name = {r.name: r for r in self.results}

        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in self.results:
            if r.error:
                self.tree.insert("", tk.END, values=(r.name, "—", "—", "—"))
            else:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        r.name,
                        f"{r.aic:.2f}",
                        f"{r.bic:.2f}",
                        f"{r.weight_percent:.2f}",
                    ),
                )

        valid_names = [r.name for r in self.results if r.error is None]
        self.plot_combo["values"] = valid_names
        if valid_names:
            self.plot_combo.current(0)
            self.plot_var.set(valid_names[0])

        best = self.results[0]
        self.status_var.set(
            f"Столбец «{column}», n={len(self.data)} (из {len(self.raw_data)}). "
            f"Лучшая модель: {best.name} (вес {best.weight_percent:.2f} %)"
        )

        if valid_names:
            self.render("pdf")

    def reset_filters(self):
        """Очищает поля обрезки и перезапускает анализ."""
        if not self.file_path:
            return
        self.start_idx.set("")
        self.end_idx.set("")
        self.analyze()

    def _on_tree_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0], "values")[0]
        valid = self.plot_combo["values"] or ()
        if name in valid:
            self.plot_var.set(name)

    def render(self, kind: str):
        name = self.plot_var.get()
        if not name or name not in self.fit_by_name:
            messagebox.showinfo("Нет выбора", "Сначала выберите распределение")
            return
        result = self.fit_by_name[name]
        
        if kind == "pdf":
            # Получаем количество бинов
            bins_val = self.bins_var.get().strip()
            if not bins_val:
                bins_val = "auto"
            
            # Если ввели число, конвертируем в int
            try:
                bins = int(bins_val)
                if bins < 2: bins = "auto" # Защита от 0 и 1
            except ValueError:
                bins = bins_val # Если ввели "auto" или что-то другое, matplotlib попытается понять
                
            fig = plot_fit(self.data, result, bins=bins)
        else:
            fig = plot_qq(self.data, result)
            
        self._show_figure(fig)

    def save_current(self):
        if self.fig is None:
            messagebox.showinfo("Нечего сохранять", "Сначала постройте график")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return
        try:
            save_fig(self.fig, path)
            self.status_var.set(f"Сохранено: {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{exc}")

    # ------------------------------------------------------------- helpers
    def _show_figure(self, fig):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        if self.fig is not None:
            plt.close(self.fig)

        self.fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.canvas_frame)
        toolbar.update()
        toolbar.pack(fill=tk.X)


def run():
    app = App()
    app.mainloop()