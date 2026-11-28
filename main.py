"""Точка входа приложения: запуск GUI."""
import sys


def main() -> int:
    from src.gui import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())