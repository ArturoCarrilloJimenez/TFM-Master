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


def test_extraer_contexto_banco_finbert():
    """Verifica la extracción semántica de oraciones completas (ventana 0 y +/- 1) con [SEP] y abreviaturas."""
    titular = "Wall Street closes higher amid corporate earnings wave"
    cuerpo = (
        "Markets rallied today driven by tech stocks. "
        "Inflation fears eased following new economic data. "
        "Goldman Sachs Inc. reported net earnings up 25% beating analysts estimates. "
        "The investment bank saw strong performance in its equities trading unit. "
        "Meanwhile retail companies suffered margin contractions. "
        "Overall trading volume remained elevated throughout the session."
    )

    # 1. Ventana 0 (foco estricto contra contagio sintáctico): solo titular + frase del banco
    texto_v0, localizado_v0 = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular,
        cuerpo=cuerpo,
        ticker_banco="GS",
        ventana_frases=0,
    )
    assert localizado_v0 is True
    assert f"{titular} [SEP] " in texto_v0
    assert "Goldman Sachs Inc. reported net earnings up 25%" in texto_v0
    assert "Inflation fears eased" not in texto_v0

    # 2. Ventana ±1: incluye frase anterior + frase del banco + frase posterior
    texto_v1, localizado_v1 = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular,
        cuerpo=cuerpo,
        ticker_banco="GS",
        ventana_frases=1,
    )
    assert localizado_v1 is True
    assert "Inflation fears eased following new economic data." in texto_v1
    assert "The investment bank saw strong performance in its equities trading unit." in texto_v1
    assert "Markets rallied today driven by tech stocks." not in texto_v1
    assert "Overall trading volume remained elevated" not in texto_v1


def test_extraer_contexto_banco_abreviaturas_y_normalizacion():
    """Verifica que la regex no rompa abreviaturas (J.P., U.S., Inc.) y normalice alias sin puntos."""
    titular = "Bank earnings beat expectations"
    cuerpo = "J.P. Morgan Chase & Co. reported profit of $3.5 billion. The U.S. lender beat estimates."
    texto, localizado = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular,
        cuerpo=cuerpo,
        ticker_banco="JPM",
        ventana_frases=0,
    )
    assert localizado is True
    assert "J.P. Morgan Chase & Co. reported profit of $3.5 billion." in texto
    assert "The U.S. lender beat estimates." not in texto


def test_extraer_contexto_banco_fallback_sin_mencion_en_cuerpo():
    """Verifica que si el banco no está en el cuerpo, caiga al titular + primera frase."""
    titular = "JPMorgan CEO Jamie Dimon speaks at annual economic forum"
    cuerpo = (
        "Global growth is facing headwinds from monetary tightening. "
        "Geopolitical tensions continue to weigh on consumer confidence."
    )

    texto_extraido, localizado = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular,
        cuerpo=cuerpo,
        ticker_banco="JPM",
        ventana_frases=1,
    )

    assert localizado is False
    assert titular in texto_extraido
    assert "Global growth is facing headwinds from monetary tightening." in texto_extraido


def test_extraer_contexto_banco_sin_cuerpo():
    """Verifica el comportamiento cuando no hay cuerpo (noticia solo titular)."""
    titular = "Bank of America raises prime lending rate to 8.5%"
    texto_extraido, localizado = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular,
        cuerpo="",
        ticker_banco="BAC",
    )

    assert localizado is False
    assert texto_extraido == titular


def test_extraer_contexto_banco_titular_condicional_y_adaptativo():
    """Verifica el gating condicional del titular, la ventana adaptativa mono/multi-ticker y extraer_frases_banco."""
    titular_ajeno = "Financial ETFs Down on Mixed Sector Earnings"
    cuerpo_mono = (
        "Broader markets opened lower. "
        "Wells Fargo fourth-quarter earnings per share of 67 cents outpaced the consensus estimate. "
        "The lender also reported higher net interest income."
    )
    # 1. Titular condicional: omite el titular porque no menciona a WFC; ventana adaptativa usa ±1 por ser mono-ticker
    txt_adapt, loc = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular_ajeno,
        cuerpo=cuerpo_mono,
        ticker_banco="WFC",
        titular_condicional=True,
        ventana_auto_multiticker=True,
    )
    assert loc is True
    assert titular_ajeno not in txt_adapt
    assert "Wells Fargo fourth-quarter earnings" in txt_adapt
    assert "The lender also reported higher net interest income." in txt_adapt

    # 2. En noticia multi-ticker, ventana_auto_multiticker fuerza ventana 0
    cuerpo_multi = (
        "Citigroup shares fell 4% after weak trading results. "
        "Wells Fargo fourth-quarter earnings per share of 67 cents outpaced the consensus estimate."
    )
    txt_multi, _ = BaseProcesadorArticulo.extraer_contexto_banco(
        titular=titular_ajeno,
        cuerpo=cuerpo_multi,
        ticker_banco="WFC",
        titular_condicional=True,
        ventana_auto_multiticker=True,
    )
    assert "Citigroup shares fell 4%" not in txt_multi
    assert "Wells Fargo fourth-quarter earnings" in txt_multi

    # 3. Extracción de lista de frases individuales
    frases = BaseProcesadorArticulo.extraer_frases_banco(titular_ajeno, cuerpo_multi, "WFC", titular_condicional=True)
    assert len(frases) == 1
    assert "Wells Fargo" in frases[0]
