import runpy


def test_main_executes_cleanly(capsys):
    """Verifica que main() se ejecute como módulo e imprima el saludo inicial."""
    runpy.run_module("main", run_name="__main__")
    captured = capsys.readouterr()
    assert "Hello from tfm-master!" in captured.out
