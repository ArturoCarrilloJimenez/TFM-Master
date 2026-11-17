"""Tests unitarios para la clase base BaseProcesadorArticulo y extracción contextual para FinBERT."""

from articulos.models.base_nlp import BaseProcesadorArticulo


def test_contiene_alias_frontera_palabra_base():
    """Verifica que la clase base controle fronteras de palabra sin falsos positivos."""
    assert not BaseProcesadorArticulo.contiene_alias("Quarterly earnings were strong", "gs")
    assert BaseProcesadorArticulo.contiene_alias("Goldman Sachs GS reported higher revenue", "gs")
    assert not BaseProcesadorArticulo.contiene_alias("Terms of the deal were not disclosed", "ms")
    assert BaseProcesadorArticulo.contiene_alias("Morgan Stanley MS upgraded the stock", "ms")


def test_resolver_cuerpo_noticia_base():
    """Verifica que la clase base resuelva el cuerpo del artículo y filtre disclaimers."""
    row = {"article": "Contenido analítico del banco.", "text": "Texto alternativo"}
    assert BaseProcesadorArticulo.resolver_cuerpo_noticia(row, "Titular") == "Contenido analítico del banco."

    row_vacio = {"article": "", "text": "   "}
    assert BaseProcesadorArticulo.resolver_cuerpo_noticia(row_vacio, "Titular") == ""


def test_extraer_oraciones_banco_finbert():
    """Verifica la extracción semántica de oraciones completas (+/- 1 ventana) para FinBERT."""
    titular = "Wall Street closes higher amid corporate earnings wave"
    cuerpo = (
        "Markets rallied today driven by tech stocks. "
        "Inflation fears eased following new economic data. "
        "Goldman Sachs reported net earnings up 25% beating analysts estimates. "
        "The investment bank saw strong performance in its equities trading unit. "
        "Meanwhile retail companies suffered margin contractions. "
        "Overall trading volume remained elevated throughout the session."
    )

    # 1. Caso con banco localizado: debe incluir la frase anterior + la frase del banco + la frase posterior
    texto_extraido, localizado = BaseProcesadorArticulo.extraer_oraciones_banco(
        titular=titular,
        cuerpo=cuerpo,
        ticker_banco="GS",
        ventana_frases=1,
    )

    assert localizado is True
    assert titular in texto_extraido
    assert "Goldman Sachs reported net earnings up 25%" in texto_extraido
    # Frase anterior incluida
    assert "Inflation fears eased following new economic data." in texto_extraido
    # Frase posterior incluida
    assert "The investment bank saw strong performance in its equities trading unit." in texto_extraido
    # Frases lejanas NO deben estar incluidas
    assert "Markets rallied today driven by tech stocks." not in texto_extraido
    assert "Overall trading volume remained elevated" not in texto_extraido


def test_extraer_oraciones_banco_fallback_sin_mencion_en_cuerpo():
    """Verifica que si el banco no está en el cuerpo, caiga al titular + primera frase."""
    titular = "JPMorgan CEO Jamie Dimon speaks at annual economic forum"
    cuerpo = (
        "Global growth is facing headwinds from monetary tightening. "
        "Geopolitical tensions continue to weigh on consumer confidence."
    )

    texto_extraido, localizado = BaseProcesadorArticulo.extraer_oraciones_banco(
        titular=titular,
        cuerpo=cuerpo,
        ticker_banco="JPM",
        ventana_frases=1,
    )

    assert localizado is False
    assert titular in texto_extraido
    assert "Global growth is facing headwinds from monetary tightening." in texto_extraido


def test_extraer_oraciones_banco_sin_cuerpo():
    """Verifica el comportamiento cuando no hay cuerpo (noticia solo titular)."""
    titular = "Bank of America raises prime lending rate to 8.5%"
    texto_extraido, localizado = BaseProcesadorArticulo.extraer_oraciones_banco(
        titular=titular,
        cuerpo="",
        ticker_banco="BAC",
    )

    assert localizado is False
    assert texto_extraido == titular
