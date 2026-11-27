"""Estándar editorial unificado de Matplotlib/Seaborn para las figuras de la memoria TFM.

Alinea la tipografía, paleta cromática institucional, tamaños de fuente y exportación
vectorial (.pdf) / alta resolución (.png 300 DPI) con `docs/memoria/main.tex`.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

COLORES_TEMA: dict[str, str] = {
    "navy": "#081C3F",
    "cyan": "#0077B6",
    "slate": "#111827",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "light_bg": "#F1F5F9",
    "teal": "#0D9488",
    "amber": "#D97706",
    "coral": "#DC2626",
}

ORDEN_BANCOS: list[str] = ["JPM", "BAC", "WFC", "C", "GS", "MS"]

PALETA_BANCOS: dict[str, str] = {
    "JPM": "#081C3F",  # Navy Primary
    "BAC": "#0077B6",  # Cyan Accent
    "WFC": "#0D9488",  # Teal
    "C": "#D97706",    # Amber
    "GS": "#DC2626",   # Coral
    "MS": "#64748B",   # Muted Slate
}

FIGSIZE_SINGLE: tuple[float, float] = (10.5, 4.2)
FIGSIZE_DOUBLE: tuple[float, float] = (11.5, 4.4)
DPI_PUBLICACION: int = 300


def aplicar_estilo_memoria() -> None:
    """Configura `rcParams` de Matplotlib con el estándar editorial único del TFM."""
    rc: dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORES_TEMA["navy"],
        "axes.titlepad": 10.0,
        "axes.labelsize": 9.5,
        "axes.labelcolor": COLORES_TEMA["slate"],
        "axes.labelweight": "semibold",
        "axes.edgecolor": COLORES_TEMA["muted"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": COLORES_TEMA["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": COLORES_TEMA["slate"],
        "ytick.color": COLORES_TEMA["slate"],
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 8.8,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": COLORES_TEMA["grid"],
        "legend.framealpha": 0.95,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI_PUBLICACION,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    mpl.rcParams.update(rc)


def guardar_figura(fig: Figure, destino_sin_ext: Path, cerrar: bool = True) -> tuple[Path, Path]:
    """Exporta una figura en formato vectorial `.pdf` y rasterizado `.png` (300 DPI).

    Args:
        fig: Instancia de `matplotlib.figure.Figure` lista para exportar.
        destino_sin_ext: Ruta base sin extensión (ej. `docs/memoria/figures/mi_grafico`).
        cerrar: Si es True, cierra la figura tras guardar para liberar memoria.

    Returns:
        Tupla `(ruta_pdf, ruta_png)` con los archivos generados.
    """
    destino_sin_ext.parent.mkdir(parents=True, exist_ok=True)
    ruta_pdf = destino_sin_ext.with_suffix(".pdf")
    ruta_png = destino_sin_ext.with_suffix(".png")
    fig.savefig(ruta_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(ruta_png, format="png", dpi=DPI_PUBLICACION, bbox_inches="tight", facecolor="white")
    if cerrar:
        plt.close(fig)
    return ruta_pdf, ruta_png
