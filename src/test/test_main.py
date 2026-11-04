"""Tests unitarios para el punto de entrada principal (src/main.py)."""

import runpy


def test_main_executes_cleanly(capsys):
    """Verifica que main() se ejecute como módulo y emita el log de inicio."""
    runpy.run_module("main", run_name="__main__")
    captured = capsys.readouterr()
    assert "Iniciando TFM-Master..." in captured.out
