"""Tests unitarios para el clasificador estructural de eventos SEC Form 8-K."""

from sec.clasificador_estructural import (
    clasificar_item_8k,
    EXHIBIT_REF_PATTERN,
    SUBSTANTIVE_METRICS_PATTERN,
    SAFE_HARBOR_PATTERN,
)


def test_clasificar_item_8k_announcement_only():
    """Valida que una carátula procedimental se categorice como announcement_only."""
    text = (
        "On October 12, 2018, Citigroup Inc. announced its results for the quarter ended September 30, 2018. "
        "A copy of the related press release, filed as Exhibit 99.1 to this Form 8-K, is incorporated herein by reference. "
        "In addition, a copy of the Quarterly Financial Data Supplement is furnished as Exhibit 99.2."
    )
    res = clasificar_item_8k("Item 2.02", "[C] Item 2.02: Results", text)
    assert res["is_announcement_only"] is True
    assert res["category"] == "announcement_only"
    assert res["has_metrics"] is False
    assert res["item_code"] == "Item 2.02" if "item_code" in res else True


def test_clasificar_item_8k_substantive_body_with_safe_harbor():
    """Valida que un cuerpo con métricas económicas pode el Safe Harbor y se marque substantive_body."""
    text = (
        "JPMorgan Chase reported 2021 first quarter net income of $14.3 billion, or $4.50 per share, "
        "compared with net income of $2.9 billion, or $0.78 per share, in the first quarter of 2020. "
        "These statements are based on the current beliefs and expectations of management and are subject to "
        "significant risks and uncertainties. Factors that could cause actual results to differ materially from those "
        "described in the forward-looking statements can be found in our Annual Report on Form 10-K."
    )
    res = clasificar_item_8k("Item 2.02", "[JPM] Item 2.02: Results", text)
    assert res["is_announcement_only"] is False
    assert res["category"] == "substantive_body"
    assert res["has_metrics"] is True
    # Comprobar que el Safe Harbor fue podado del texto para FinBERT
    assert "Private Securities Litigation" not in res["texto_limpio_finbert"]
    assert "Annual Report on Form 10-K" not in res["texto_limpio_finbert"]
    assert "$14.3 billion" in res["texto_limpio_finbert"]


def test_clasificar_item_8k_regulatory_penalty():
    """Valida la detección de penalizaciones y acuerdos regulatorios sustantivos."""
    text = (
        "JPMorgan Chase Bank entered into a consent order with the Commodity Futures Trading Commission "
        "and agreed to pay a $100 million civil money penalty in connection with credit default swap trading."
    )
    res = clasificar_item_8k("Item 8.01", "[JPM] Item 8.01: Events", text)
    assert res["is_announcement_only"] is False
    assert res["category"] == "substantive_body"
    assert res["has_metrics"] is True
    assert "$100 million" in res["texto_limpio_finbert"]


def test_clasificar_item_8k_empty_text():
    """Valida el manejo de textos vacíos."""
    res = clasificar_item_8k("Item 2.02", "[C] Item 2.02", "")
    assert res["is_announcement_only"] is True
    assert res["category"] == "announcement_only"
    assert res["has_metrics"] is False
    assert res["words_substantive"] == 0
    assert res["texto_limpio_finbert"] == "[C] Item 2.02"


def test_regex_patterns_direct():
    """Valida la cobertura directa de las expresiones regulares clave."""
    assert EXHIBIT_REF_PATTERN.search("filed as exhibit 99.1") is not None
    assert EXHIBIT_REF_PATTERN.search("incorporated herein by reference") is not None
    assert SUBSTANTIVE_METRICS_PATTERN.search("net income rose to $4.5 billion") is not None
    assert SUBSTANTIVE_METRICS_PATTERN.search("diluted eps was $1.75") is not None
    assert SAFE_HARBOR_PATTERN.search("Private Securities Litigation Reform Act of 1995") is not None
