"""Generador reproducible de figuras editoriales para la memoria del TFM (PDF vectorial + PNG 300 DPI).

Reutiliza `shared.plot_style`, `get_compactor()` y `get_storage_repository()` para garantizar
uniformidad visual en toda la memoria y adherencia a los principios KISS y DRY.
"""

import asyncio
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dependencies.dependencies import get_compactor, get_storage_repository  # noqa: E402
from shared.logging import Logger  # noqa: E402
from shared.plot_style import (  # noqa: E402
    COLORES_TEMA,
    FIGSIZE_DOUBLE,
    FIGSIZE_SINGLE,
    ORDEN_BANCOS,
    PALETA_BANCOS,
    aplicar_estilo_memoria,
    guardar_figura,
)

logger = Logger("GenerarFigurasMemoria")

FIGURES_DIR: Path = PROJ_ROOT / "docs" / "memoria" / "figures"
LOCAL_MULTISOURCE_BRONZE: Path = PROJ_ROOT / "src" / "data" / "multisource_bancos_2010_2023.parquet"

# Matriz anual auditada en src/notebooks/articulos/huggingface/huggingface_2010_2023_EDA.ipynb
FNSPID_V1_ANUAL: dict[int, dict[str, int]] = {
    2010: {"BAC": 0, "C": 0, "GS": 0, "JPM": 0, "MS": 602, "WFC": 0},
    2011: {"BAC": 0, "C": 0, "GS": 0, "JPM": 0, "MS": 527, "WFC": 2},
    2012: {"BAC": 0, "C": 0, "GS": 0, "JPM": 0, "MS": 365, "WFC": 309},
    2013: {"BAC": 0, "C": 0, "GS": 0, "JPM": 0, "MS": 257, "WFC": 285},
    2014: {"BAC": 88, "C": 101, "GS": 0, "JPM": 0, "MS": 375, "WFC": 222},
    2015: {"BAC": 310, "C": 767, "GS": 0, "JPM": 0, "MS": 639, "WFC": 419},
    2016: {"BAC": 323, "C": 943, "GS": 746, "JPM": 0, "MS": 619, "WFC": 2133},
    2017: {"BAC": 348, "C": 1314, "GS": 972, "JPM": 0, "MS": 759, "WFC": 1928},
    2018: {"BAC": 220, "C": 1077, "GS": 805, "JPM": 499, "MS": 713, "WFC": 1095},
    2019: {"BAC": 303, "C": 805, "GS": 563, "JPM": 1724, "MS": 551, "WFC": 833},
    2020: {"BAC": 219, "C": 761, "GS": 1028, "JPM": 660, "MS": 918, "WFC": 1182},
    2021: {"BAC": 0, "C": 0, "GS": 1436, "JPM": 0, "MS": 889, "WFC": 857},
    2022: {"BAC": 0, "C": 1125, "GS": 1429, "JPM": 0, "MS": 1018, "WFC": 850},
    2023: {"BAC": 0, "C": 357, "GS": 1751, "JPM": 0, "MS": 1226, "WFC": 1186},
}


def _fmt_miles(x: float, _pos: int) -> str:
    """Formatea valores enteros del eje Y con separador de miles en español."""
    return f"{int(round(x)):,}".replace(",", ".")


async def _cargar_parquet_df(s3_key: str, local_path: Path | None = None) -> pd.DataFrame:
    """Carga un Parquet desde disco local o Cloudflare R2 usando `ParquetCompactor`."""
    compactor = get_compactor()
    if local_path is not None and local_path.exists():
        raw_bytes = local_path.read_bytes()
    else:
        repo = get_storage_repository()
        try:
            raw_bytes = await repo.get(s3_key)
        finally:
            await repo.close()
    return pd.DataFrame(compactor.decompact(raw_bytes))


def _plot_barras_anuales_bancos(
    df_anual: pd.DataFrame,
    titulo: str,
    ylabel: str,
    destino_sin_ext: Path,
) -> None:
    """Dibuja un gráfico de barras apiladas anual con paleta fija por banco y estilo editorial."""
    cols = [b for b in ORDEN_BANCOS if b in df_anual.columns]
    datos = df_anual[cols]
    colores = [PALETA_BANCOS[b] for b in cols]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    datos.plot(
        kind="bar",
        stacked=True,
        color=colores,
        edgecolor="white",
        linewidth=0.5,
        width=0.72,
        ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel("Año")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_miles))
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        title="Entidad (Ticker)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
    )
    fig.tight_layout()
    pdf_p, png_p = guardar_figura(fig, destino_sin_ext)
    logger.info(f"Generada figura: {pdf_p.name} + {png_p.name}")


def _familia_anexo(ex_type: str) -> str:
    """Agrupa códigos `EX-*` de la SEC en familias editoriales legibles."""
    t = str(ex_type).upper()
    if t.startswith("EX-99"):
        return "EX-99* (Comunicados / Resultados)"
    if t.startswith(("EX-5", "EX-8")):
        return "EX-5* / EX-8* (Legal / Fiscal)"
    if t.startswith(("EX-4", "EX-1.")):
        return "EX-4* / EX-1* (Instrumentos / Emisión)"
    return "Otros anexos EX-*"


def generar_figuras_bronze_noticias(df_multi: pd.DataFrame) -> None:
    """Regenera las figuras comparativas de cobertura anual de noticias (FNSPID v1 vs Multisource)."""
    df_fnspid = pd.DataFrame.from_dict(FNSPID_V1_ANUAL, orient="index")
    df_fnspid.index.name = "year"
    _plot_barras_anuales_bancos(
        df_anual=df_fnspid,
        titulo="Cobertura histórica anual por entidad — FNSPID v1 (2010–2023)",
        ylabel="Número de noticias",
        destino_sin_ext=FIGURES_DIR / "bronze_fnspid_v1_cobertura_anual",
    )

    dt = pd.to_datetime(df_multi["datetime"], utc=True, errors="coerce")
    df_tmp = df_multi.assign(year=dt.dt.year)
    matriz_multi = df_tmp.groupby(["year", "ticker"]).size().unstack(fill_value=0)
    _plot_barras_anuales_bancos(
        df_anual=matriz_multi,
        titulo="Cobertura histórica anual por entidad — Multisource consolidado (2010–2023)",
        ylabel="Número de noticias",
        destino_sin_ext=FIGURES_DIR / "bronze_multisource_cobertura_anual",
    )


def generar_figura_sec_exhibits(df_8k: pd.DataFrame, df_exhibits: pd.DataFrame) -> None:
    """Regenera la figura dual de contraste textual Item 2.02 vs EX-99* y evolución anual de anexos."""
    df_8k = df_8k.copy()
    df_exhibits = df_exhibits.copy()

    df_8k["word_count"] = df_8k["content"].fillna("").astype(str).str.split().str.len().clip(lower=1)
    df_exhibits["word_count"] = df_exhibits["content"].fillna("").astype(str).str.split().str.len().clip(lower=1)
    df_exhibits["has_error"] = df_exhibits["error"].fillna("").astype(str).str.strip().ne("")
    df_exhibits["year"] = pd.to_datetime(df_exhibits["filing_date"], errors="coerce").dt.year

    item_202_words = df_8k.loc[df_8k["item"] == "Item 2.02", "word_count"].to_numpy()
    ex99_mask = df_exhibits["exhibit_type"].fillna("").str.startswith("EX-99") & (~df_exhibits["has_error"])
    ex99_words = df_exhibits.loc[ex99_mask, "word_count"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Panel izquierdo: Boxplot en escala logarítmica
    bp = axes[0].boxplot(
        [item_202_words, ex99_words],
        tick_labels=["Carátula Item 2.02\n(8-K base)", "Anexo EX-99*\n(Comunicado real)"],
        patch_artist=True,
        widths=0.48,
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": COLORES_TEMA["muted"], "linewidth": 1.0},
        capprops={"color": COLORES_TEMA["muted"], "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": COLORES_TEMA["muted"],
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
    )
    for patch, color in zip(bp["boxes"], [COLORES_TEMA["coral"], COLORES_TEMA["cyan"]]):
        patch.set_facecolor(color)
        patch.set_edgecolor(COLORES_TEMA["navy"])
        patch.set_linewidth(0.9)

    axes[0].set_yscale("log")
    axes[0].set_title("Longitud textual: Carátula Item 2.02 vs. EX-99*")
    axes[0].set_ylabel("Número de palabras (escala log10)")

    # Panel derecho: Evolución anual de anexos por familia
    df_exhibits["familia"] = df_exhibits["exhibit_type"].apply(_familia_anexo)
    orden_familias = [
        "EX-99* (Comunicados / Resultados)",
        "EX-5* / EX-8* (Legal / Fiscal)",
        "EX-4* / EX-1* (Instrumentos / Emisión)",
        "Otros anexos EX-*",
    ]
    colores_fam = [
        COLORES_TEMA["navy"],
        COLORES_TEMA["cyan"],
        COLORES_TEMA["teal"],
        COLORES_TEMA["muted"],
    ]
    matriz_exh = pd.crosstab(df_exhibits["year"].dropna().astype(int), df_exhibits["familia"])
    cols_fam = [f for f in orden_familias if f in matriz_exh.columns]
    matriz_exh[cols_fam].plot(
        kind="bar",
        stacked=True,
        color=colores_fam[: len(cols_fam)],
        edgecolor="white",
        linewidth=0.5,
        width=0.72,
        ax=axes[1],
    )
    axes[1].set_title("Evolución anual de anexos 8-K por familia (2010–2023)")
    axes[1].set_xlabel("Año")
    axes[1].set_ylabel("Número de anexos")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_fmt_miles))
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Familia de anexo", loc="upper right", fontsize=7.8)

    fig.tight_layout(w_pad=2.2)
    pdf_p, png_p = guardar_figura(fig, FIGURES_DIR / "bronze_sec_8k_vs_exhibits")
    logger.info(f"Generada figura: {pdf_p.name} + {png_p.name}")


def generar_figura_gatekeeper_silver(df_silver: pd.DataFrame) -> None:
    """Genera la figura dual del Gatekeeper Semántico: auditoría v1 vs. v2 y distribución temática Silver."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Panel izquierdo: Auditoría iterativa v1 vs. v2 (métricas en %)
    metricas = [
        "Precisión\n(PPV, N=60)",
        "Pureza descarte\n(NPV, N=60)",
        "Exactitud global\n(N=60)",
        "Anclaje explícito\ndel banco",
    ]
    v1_vals = [60.0, 83.3, 71.7, 80.05]
    v2_vals = [83.3, 93.3, 88.3, 97.98]

    x = list(range(len(metricas)))
    width = 0.35
    bars_v1 = axes[0].bar(
        [i - width / 2 for i in x],
        v1_vals,
        width=width,
        label="Gatekeeper v1 (inicial)",
        color=COLORES_TEMA["muted"],
        edgecolor="white",
        linewidth=0.6,
    )
    bars_v2 = axes[0].bar(
        [i + width / 2 for i in x],
        v2_vals,
        width=width,
        label="Gatekeeper v2 (anclaje + reglas)",
        color=COLORES_TEMA["cyan"],
        edgecolor="white",
        linewidth=0.6,
    )
    for bars in (bars_v1, bars_v2):
        for b in bars:
            h = b.get_height()
            axes[0].text(
                b.get_x() + b.get_width() / 2,
                h + 1.4,
                f"{h:.1f}%".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=COLORES_TEMA["navy"],
                fontweight="semibold",
            )

    axes[0].set_title("Auditoría iterativa del Gatekeeper (v1 vs. v2)")
    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_ylim(0, 122)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metricas, fontsize=8.0)
    axes[0].legend(loc="upper left", ncol=1, fontsize=7.4, frameon=True, framealpha=0.92)

    # Panel derecho: Distribución temática final en Capa Silver (27.350 noticias útiles)
    etiquetas_cat = {
        "m&a": "M&A y Reestructuración",
        "earnings": "Resultados (Earnings)",
        "legal": "Regulación y Litigios",
        "macro": "Macroeconomía y Fed",
        "governance": "Gobierno Corporativo",
        "credit": "Riesgo Crediticio",
    }
    conteos = df_silver["category"].value_counts()
    cats_ord = list(conteos.index)
    vals_ord = [int(conteos[c]) for c in cats_ord]
    labels_ord = [etiquetas_cat.get(c, str(c)) for c in cats_ord]
    total_silver = max(len(df_silver), 1)

    colores_cat = [
        COLORES_TEMA["navy"],
        COLORES_TEMA["cyan"],
        COLORES_TEMA["teal"],
        COLORES_TEMA["amber"],
        COLORES_TEMA["coral"],
        COLORES_TEMA["muted"],
    ]
    y_pos = list(range(len(cats_ord)))
    bars_cat = axes[1].barh(
        y_pos,
        vals_ord,
        color=colores_cat[: len(cats_ord)],
        edgecolor="white",
        linewidth=0.6,
        height=0.62,
    )
    axes[1].invert_yaxis()
    max_val = max(vals_ord) if vals_ord else 1
    for b, v in zip(bars_cat, vals_ord):
        pct = v / total_silver * 100.0
        txt_num = f"{v:,}".replace(",", ".")
        txt_pct = f"{pct:.1f}%".replace(".", ",")
        axes[1].text(
            v + max_val * 0.02,
            b.get_y() + b.get_height() / 2,
            f"{txt_num} ({txt_pct})",
            va="center",
            ha="left",
            fontsize=7.6,
            color=COLORES_TEMA["navy"],
            fontweight="semibold",
        )

    axes[1].set_title(f"Distribución temática en Capa Silver (N = {_fmt_miles(total_silver, 0)})")
    axes[1].set_xlabel("Número de noticias útiles")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels_ord, fontsize=8.2)
    axes[1].set_xlim(0, max_val * 1.38)
    axes[1].xaxis.set_major_formatter(FuncFormatter(_fmt_miles))
    axes[1].grid(axis="x", linestyle="--", alpha=0.45)
    axes[1].grid(axis="y", visible=False)

    fig.tight_layout(w_pad=2.4)
    pdf_p, png_p = guardar_figura(fig, FIGURES_DIR / "silver_gatekeeper_auditoria")
    logger.info(f"Generada figura: {pdf_p.name} + {png_p.name}")


async def main() -> None:
    """Orquesta la regeneración uniforme de todas las figuras de la memoria."""
    aplicar_estilo_memoria()
    df_multi = await _cargar_parquet_df(
        "raw/huggingface/multisource_bancos_2010_2023.parquet",
        local_path=LOCAL_MULTISOURCE_BRONZE,
    )
    generar_figuras_bronze_noticias(df_multi)

    df_8k = await _cargar_parquet_df("raw/sec/8k-2010-2023.parquet")
    df_exhibits = await _cargar_parquet_df("raw/sec/8k-exhibits-2010-2023.parquet")
    generar_figura_sec_exhibits(df_8k, df_exhibits)

    df_silver = await _cargar_parquet_df("processed/huggingface/multisource_bancos_2010_2023.parquet")
    generar_figura_gatekeeper_silver(df_silver)
    logger.info("Todas las figuras editoriales de la memoria han sido regeneradas.")


if __name__ == "__main__":
    asyncio.run(main())
