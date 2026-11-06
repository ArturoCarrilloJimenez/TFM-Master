"""Tests unitarios para la clase CategorizadorArticulos y sus heurísticas de saliencia semántica."""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from articulos.models.categorizador import (
    CategorizadorArticulos,
    get_optimal_workers,
    is_running_in_lambda,
)


def test_contiene_alias_frontera_palabra():
    """Verifica que contiene_alias respete las fronteras de palabra sin colisiones accidentales."""
    # 'gs' no debe disparar en 'earnings' o 'things'
    assert not CategorizadorArticulos.contiene_alias("Quarterly earnings were strong", "gs")
    assert CategorizadorArticulos.contiene_alias("Goldman Sachs GS reported higher revenue", "gs")
    # 'c' en Citigroup
    assert not CategorizadorArticulos.contiene_alias("Company executive spoke today", "c")
    assert CategorizadorArticulos.contiene_alias("Citigroup announces restructuring", "citigroup")


def test_resolver_cuerpo_noticia():
    """Verifica la resolución de cuerpos, filtrando disclaimers y redundancias."""
    # 1. Prioriza article si existe
    row1 = {"article": "Contenido detallado del artículo.", "text": "Texto secundario"}
    assert CategorizadorArticulos.resolver_cuerpo_noticia(row1, "Titular") == "Contenido detallado del artículo."

    # 2. Fallback a text si article está vacío
    row2 = {"article": "", "text": "Texto alternativo presente."}
    assert CategorizadorArticulos.resolver_cuerpo_noticia(row2, "Titular") == "Texto alternativo presente."

    # 3. Descarta si es igual al titular
    row3 = {"article": "Mismo titular", "text": "Mismo titular"}
    assert CategorizadorArticulos.resolver_cuerpo_noticia(row3, "Mismo titular") == ""

    # 4. Descarta disclaimers de opinión legales cortos
    row4 = {"article": "The views and opinions expressed in this piece do not necessarily reflect the official policy."}
    assert CategorizadorArticulos.resolver_cuerpo_noticia(row4, "Titular") == ""


def test_extraer_fragmento_relevante_boilerplate():
    """Verifica la eliminación de boilerplate sectorial."""
    cuerpo = (
        "Regional banking has been volatile this quarter. "
        "JPMorgan Chase raised its dividend by 10 percent following stress test results.\n\n"
        "JPMorgan is a part of the finance sector, which includes Bank of America and Wells Fargo."
    )
    frag = CategorizadorArticulos.extraer_fragmento_relevante("Titular", cuerpo, "JPM")
    assert "raised its dividend by 10 percent" in frag
    assert "part of the finance sector" not in frag


def test_extraer_fragmento_y_presencia():
    """Verifica que extraer_fragmento_y_presencia reporte correctamente la presencia del banco."""
    cuerpo = "Economic update. Wells Fargo reported solid retail deposit growth this year."
    frag, presente = CategorizadorArticulos.extraer_fragmento_y_presencia("Update", cuerpo, "WFC")
    assert presente is True
    assert "Wells Fargo" in frag

    cuerpo_ajeno = "Tesla shares fell amid broader automotive market declines."
    frag_ajeno, presente_ajeno = CategorizadorArticulos.extraer_fragmento_y_presencia("Update", cuerpo_ajeno, "WFC")
    assert presente_ajeno is False


def test_es_rol_analista_tercero():
    """Verifica la detección de rol sintáctico de analista (banco autor, no sujeto corporativo)."""
    titular_tercero = "Apple shares jump 4% after bullish upgrade"
    cuerpo_tercero = "JPMorgan analysts issued a positive note to clients regarding iPhone demand."
    assert CategorizadorArticulos.es_rol_analista_tercero(titular_tercero, cuerpo_tercero, "JPM")

    titular_sujeto = "JPMorgan Chase reports record Q2 net income"
    cuerpo_sujeto = "The bank achieved record profits across its consumer and investment banking divisions."
    assert not CategorizadorArticulos.es_rol_analista_tercero(titular_sujeto, cuerpo_sujeto, "JPM")


@patch("articulos.models.categorizador.SentenceTransformer")
def test_clasificar_noticia_y_dataset(mock_st_cls):
    """Verifica la inferencia vectorial mockeada para clasificar_noticia y clasificar_dataset."""
    mock_instance = MagicMock()
    mock_st_cls.return_value = mock_instance

    # Mock de codificación: vectores unitarios de dimensión 4
    def fake_encode(texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.ones((len(texts), 4)) / 2.0

    mock_instance.encode.side_effect = fake_encode

    categories_test = {
        "earnings": ["earnings report", "quarterly results"],
        "ruido": ["unrelated noise", "random gossip"],
    }

    model = CategorizadorArticulos(categories=categories_test)

    # 1. Probar clasificar_noticia individual
    res = model.clasificar_noticia(
        titular="JPMorgan Chase reports record quarterly profits",
        cuerpo="The bank exceeded expectations.",
        ticker_banco="JPM",
    )
    assert "categoria" in res
    assert "confianza" in res
    assert "is_ruido" in res
    assert isinstance(res["is_ruido"], bool)

    # 2. Probar clasificar_dataset masivo
    df = pd.DataFrame([
        {
            "ticker": "JPM",
            "date": "2023-01-15",
            "title": "JPMorgan Chase reports record quarterly profits",
            "article": "Detailed article text for JPMorgan.",
        },
        {
            "ticker": "BAC",
            "date": "2023-01-16",
            "title": "Bank of America expands retail branches",
            "article": "",
        },
    ])

    results = model.clasificar_dataset(df, batch_size=2)
    assert len(results) == 2
    assert results[0]["ticker"] == "JPM"
    assert results[0]["title"] == "JPMorgan Chase reports record quarterly profits"
    assert "process_text" in results[0]
    assert "category" in results[0]
    assert "is_ruido" in results[0]


def test_lambda_detection_and_optimal_workers(monkeypatch):
    """Verifica la detección automática del entorno AWS Lambda vs entorno local."""
    # 1. En entorno local estándar (sin variables de Lambda)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("LAMBDA_TASK_ROOT", raising=False)
    assert not is_running_in_lambda()
    workers_local = get_optimal_workers()
    assert workers_local >= 1
    assert workers_local <= 4

    # Override explícito
    assert get_optimal_workers(max_workers=2) == 2

    # 2. En entorno simulado de AWS Lambda
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "tfm-scraper-sentiment")
    assert is_running_in_lambda()
    assert get_optimal_workers() == 1


def test_clasificar_dataset_preserva_todos_los_campos_y_anexa_predicciones():
    """Verifica que el modelo sea agnóstico al esquema: preserva 100% de los campos de entrada y anexa sus predicciones."""
    model = CategorizadorArticulos()
    records = [
        {
            "ticker": "JPM",
            "company_name": "JPMorgan Chase & Co.",
            "cik": "0000019617",
            "date": "2023-01-15",
            "datetime": "2023-01-15 10:00:00+00:00",
            "title": "JPMorgan reports earnings beat",
            "article": "JPMorgan profit jumped 20% in the last quarter.",
            "text": "JPMorgan reports earnings beat\nJPMorgan profit jumped 20% in the last quarter.",
            "publisher": "SentARL",
            "url": "https://example.com/news/123",
            "source_dataset": "fnspid_news",
            "ticker_count": 1,
            "is_multisubject": False,
        }
    ]

    results = model.clasificar_dataset(records, batch_size=1)
    assert len(results) == 1
    res = results[0]

    # Todos los campos originales se preservan de forma inocua por el modelo
    for k, v in records[0].items():
        assert res[k] == v

    # Columnas del modelo inyectadas
    assert "process_text" in res
    assert "category" in res
    assert "confianza" in res
    assert "is_ruido" in res


def test_etl_categorizacion_poda_esquema_silver():
    """Verifica que el orquestador ETL descarte únicamente las columnas crudas redundantes para Silver."""
    from articulos.etl_categorizacion import COLUMNAS_DESCARTAR_SILVER

    row = {
        "ticker": "JPM",
        "company_name": "JPMorgan Chase & Co.",
        "text": "full text",
        "publisher": "SentARL",
        "url": "https://example.com",
        "source_dataset": "fnspid",
        "category": "Earnings",
        "is_ruido": False,
    }

    clean_row = {k: v for k, v in row.items() if k not in COLUMNAS_DESCARTAR_SILVER}
    assert "text" not in clean_row
    assert "publisher" not in clean_row
    assert "url" not in clean_row
    assert "source_dataset" not in clean_row
    assert clean_row["ticker"] == "JPM"
    assert clean_row["category"] == "Earnings"
    assert clean_row["is_ruido"] is False
