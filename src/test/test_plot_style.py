"""Pruebas unitarias para el módulo compartido de estilo editorial Matplotlib."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from shared.plot_style import (
    COLORES_TEMA,
    DPI_PUBLICACION,
    ORDEN_BANCOS,
    PALETA_BANCOS,
    aplicar_estilo_memoria,
    guardar_figura,
)


def test_constantes_editoriales_bancos() -> None:
    """Verifica que la paleta cromática cubra exactamente los 6 bancos en orden canónico."""
    assert ORDEN_BANCOS == ["JPM", "BAC", "WFC", "C", "GS", "MS"]
    assert set(PALETA_BANCOS.keys()) == set(ORDEN_BANCOS)
    assert DPI_PUBLICACION == 300


def test_aplicar_estilo_memoria() -> None:
    """Comprueba que `aplicar_estilo_memoria` configure rcParams institucionales."""
    aplicar_estilo_memoria()
    assert mpl.rcParams["axes.titlecolor"] == COLORES_TEMA["navy"]
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["savefig.dpi"] == DPI_PUBLICACION
    assert mpl.rcParams["pdf.fonttype"] == 42


def test_guardar_figura_crea_pdf_y_png(tmp_path: Path) -> None:
    """Verifica que `guardar_figura` cree directorios padre y exporte `.pdf` y `.png`."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    destino = tmp_path / "subdir" / "figura_prueba"
    ruta_pdf, ruta_png = guardar_figura(fig, destino, cerrar=True)

    assert ruta_pdf.exists()
    assert ruta_png.exists()
    assert ruta_pdf.suffix == ".pdf"
    assert ruta_png.suffix == ".png"
    assert ruta_pdf.stat().st_size > 0
    assert ruta_png.stat().st_size > 0
