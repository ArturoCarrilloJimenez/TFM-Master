"""Módulo de modelo y clasificación semántica de artículos financieros."""

import re
import time
from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from articulos.categories import CATEGORIES
from shared.logging import Logger
from shared.tickets import (
    ALIAS_BANCOS,
    ANALYST_ROLE_TERMS,
    BOILERPLATE_SECTOR_PATTERNS,
    MACRO_KEYWORDS,
)

logger = Logger("CategorizadorArticulos")


class CategorizadorArticulos:
    """Clasificador semántico vectorial de noticias financieras con desambiguación de entidad y mitigación de ticker bleed."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        categories: dict[str, list[str]] | None = None,
    ):
        """Inicializa el Bi-Encoder y precalcula los centroides semánticos normalizados por categoría."""
        self.device = device
        self.categories_dict = categories or CATEGORIES
        logger.info(f"Cargando Bi-Encoder ({model_name}) en dispositivo [{self.device}]...")
        t0 = time.time()
        self.bi_model = SentenceTransformer(model_name, device=self.device)
        self._normalizar_matriz_centroides()
        logger.info(
            f"Modelo y centroides normalizados inicializados en {time.time() - t0:.2f} s. "
            f"Matriz de categorías: {self.matriz_centroides.shape}"
        )

    def _normalizar_matriz_centroides(self) -> None:
        """Precalcula y normaliza los centroides en espacio euclídeo para inferencia matricial O(1)."""
        centroids: dict[str, np.ndarray] = {}
        for cat, phrases in self.categories_dict.items():
            embs = self.bi_model.encode(phrases, normalize_embeddings=True)
            c = np.mean(embs, axis=0)
            centroids[cat] = c / np.linalg.norm(c)

        self.matriz_centroides = np.array(list(centroids.values()))
        self.nombres_cats = list(centroids.keys())

    @staticmethod
    def contiene_alias(texto: str, alias: str) -> bool:
        """Frontera de palabra regex estricta para evitar colisiones léxicas accidentales."""
        return bool(re.search(r"\b" + re.escape(alias) + r"\b", texto.lower()))

    @staticmethod
    def resolver_cuerpo_noticia(row: pd.Series | dict[str, Any], titular: str) -> str:
        """Resuelve el cuerpo del artículo de forma robusta y generalista:

        1. Prioriza 'article' (cuerpo extenso).
        2. Si 'article' está vacío, hace fallback a 'text' o 'summary'.
        3. Descarta redundancias si el texto es idéntico al titular o disclaimers legales.
        """
        art = str(row.get("article") or "").strip()
        if art and art.lower() not in ["none", "nan", ""]:
            candidato = art
        else:
            candidato = str(row.get("text") or row.get("summary") or "").strip()

        if (
            not candidato
            or candidato.lower() in ["none", "nan", ""]
            or candidato.strip() == titular.strip()
        ):
            return ""

        if len(candidato) < 160 and any(
            w in candidato.lower()
            for w in ["views and opinions", "do not necessarily reflect", "terms not disclosed"]
        ):
            return ""

        return candidato

    @classmethod
    def extraer_fragmento_relevante(
        cls, titular: str, cuerpo: str, ticker_banco: str, max_chars: int = 600
    ) -> str:
        """Extrae el fragmento guiado donde el banco es mencionado, evitando párrafos de boilerplate sectorial."""
        if not cuerpo or cuerpo.lower() in ["none", "nan", ""]:
            return ""

        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])
        parrafos = [p.strip() for p in re.split(r"\n+|, \n+", cuerpo) if len(p.strip()) > 30]

        # 1. Buscar párrafos legítimos donde aparezca el banco con frontera de palabra
        for p in parrafos:
            p_lower = p.lower()
            if any(bp in p_lower for bp in BOILERPLATE_SECTOR_PATTERNS):
                continue
            if any(cls.contiene_alias(p_lower, a) for a in aliases):
                return p[:max_chars]

        # 2. Si no aparece en párrafos limpios, limpiar boilerplate del cuerpo
        p_limpio = cuerpo
        for bp in BOILERPLATE_SECTOR_PATTERNS:
            p_limpio = re.sub(bp + r".*?(\n|$)", "", p_limpio, flags=re.IGNORECASE)

        if any(cls.contiene_alias(p_limpio.lower(), a) for a in aliases):
            return p_limpio[:max_chars].strip()

        # 3. Fallback: Mantener el inicio del texto limpio si el cuerpo existe
        return p_limpio[:max_chars].strip() if p_limpio.strip() else cuerpo[:max_chars].strip()

    @classmethod
    def es_rol_analista_tercero(cls, titular: str, cuerpo: str, ticker_banco: str) -> bool:
        """Desambiguación sintáctica generalista:

        Detecta si el banco NO es el sujeto corporativo de la noticia, sino únicamente el analista
        que emite una opinión o recomendación sobre un tercero o divisa.
        """
        t_lower = titular.lower()
        c_lower = cuerpo.lower()
        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])

        if not any(cls.contiene_alias(t_lower, a) for a in aliases):
            for a in aliases:
                for term in ANALYST_ROLE_TERMS:
                    if (
                        f"{a} {term}" in c_lower
                        or f"{term} at {a}" in c_lower
                        or f"{a}'s {term}" in c_lower
                        or f"{a} reiterated" in c_lower
                    ):
                        return True
        else:
            if "note to clients" in c_lower and any(
                term in c_lower for term in ["bullish conviction", "forecast", "maintains conviction"]
            ):
                return True

        return False

    def clasificar_noticia(self, titular: str, cuerpo: str, ticker_banco: str) -> dict[str, Any]:
        """Clasifica un artículo individual aplicando desambiguación, embeddings y umbrales adaptativos."""
        t_lower = titular.lower()
        fragmento = self.extraer_fragmento_relevante(titular, cuerpo, ticker_banco)
        c_lower = fragmento.lower() if fragmento else ""
        aliases = ALIAS_BANCOS.get(ticker_banco, [ticker_banco.lower()])

        # 1. Saliencia de Entidad
        es_macro = any(k in t_lower for k in MACRO_KEYWORDS) or any(
            k in c_lower[:350] for k in MACRO_KEYWORDS
        )
        banco_en_titular = any(self.contiene_alias(t_lower, a) for a in aliases)
        banco_en_entradilla = (
            any(self.contiene_alias(c_lower[:350], a) for a in aliases) if fragmento else False
        )
        banco_prominente = banco_en_titular or banco_en_entradilla

        # 2. Desambiguación de Rol (Banco como analista de un tercero)
        rol_analista = self.es_rol_analista_tercero(titular, fragmento or cuerpo, ticker_banco)

        # 3. Vectorización
        emb_t = self.bi_model.encode(titular, normalize_embeddings=True)
        es_noticia_corta = not bool(cuerpo and cuerpo.strip())

        if fragmento:
            emb_c = self.bi_model.encode(fragmento, normalize_embeddings=True)
            emb_doc = 0.65 * emb_t + 0.35 * emb_c
            emb_doc = emb_doc / np.linalg.norm(emb_doc)
        else:
            emb_doc = emb_t

        # 4. Similitud Coseno contra Prototipos
        sims = np.dot(self.matriz_centroides, emb_doc)
        idx_max = int(np.argmax(sims))
        score = float(sims[idx_max])
        cat = self.nombres_cats[idx_max]

        # 5. Umbrales Adaptativos y Anti-Bleed
        if es_noticia_corta:
            tau = 0.30 if (banco_en_titular or es_macro) else 0.38
        else:
            tau = 0.38 if (banco_prominente or es_macro) else 0.48

        if rol_analista:
            es_ruido = True
        elif not es_macro and not banco_prominente:
            es_ruido = True
        else:
            es_ruido = (cat.lower() == "ruido" or "ruido" in cat.lower()) or (score < tau)

        return {
            "categoria": cat,
            "confianza": round(score * 100, 2),
            "is_ruido": es_ruido,
            "score": score,
        }

    def clasificar_dataset(
        self, data: list[dict[str, Any]] | pd.DataFrame, batch_size: int = 256
    ) -> list[dict[str, Any]]:
        """Ejecuta inferencia semántica matricial de alto rendimiento por lotes sobre una lista de diccionarios o DataFrame."""
        records: list[dict[str, Any]] = (
            data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else data
        )
        n_rows = len(records)
        t_start = time.time()
        logger.info(f"Iniciando clasificación masiva de {n_rows:,} registros por lotes (batch_size={batch_size})...")

        # 1. Extracción y análisis de saliencia en memoria
        titulares: list[str] = []
        fragmentos: list[str] = []
        has_fragmento_mask: list[bool] = []
        es_corta_list: list[bool] = []
        es_macro_list: list[bool] = []
        banco_prominente_list: list[bool] = []
        banco_en_titular_list: list[bool] = []
        rol_analista_list: list[bool] = []
        tickers: list[str] = []
        dates: list[str] = []

        for row in records:
            t = str(row.get("title") or "").strip()
            c = self.resolver_cuerpo_noticia(row, t)
            ticker = str(row.get("ticker") or "").strip()
            frag = self.extraer_fragmento_relevante(t, c, ticker, max_chars=180)

            t_lower = t.lower()
            c_lower = frag.lower() if frag else ""
            aliases = ALIAS_BANCOS.get(ticker, [ticker.lower()])

            es_macro = any(k in t_lower for k in MACRO_KEYWORDS) or any(
                k in c_lower[:350] for k in MACRO_KEYWORDS
            )
            banco_en_titular = any(self.contiene_alias(t_lower, a) for a in aliases)
            banco_en_entradilla = (
                any(self.contiene_alias(c_lower[:350], a) for a in aliases) if frag else False
            )
            banco_prominente = banco_en_titular or banco_en_entradilla
            rol_analista = self.es_rol_analista_tercero(t, frag or c, ticker)
            es_noticia_corta = not bool(c and c.strip())

            titulares.append(t)
            fragmentos.append(frag)
            has_fragmento_mask.append(bool(frag))
            es_corta_list.append(es_noticia_corta)
            es_macro_list.append(es_macro)
            banco_prominente_list.append(banco_prominente)
            banco_en_titular_list.append(banco_en_titular)
            rol_analista_list.append(rol_analista)
            tickers.append(ticker)
            raw_date = str(row.get("date") or row.get("datetime") or "")
            dates.append(raw_date[:10])

        # 2. Vectorización masiva de titulares
        logger.info(f"Vectorizando {n_rows:,} titulares por lotes...")
        embs_t = self.bi_model.encode(
            titulares,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # 3. Vectorización de fragmentos no vacíos
        frag_indices = [i for i, hf in enumerate(has_fragmento_mask) if hf]
        frags_to_encode = [fragmentos[i] for i in frag_indices]

        embs_doc = embs_t.copy()
        if frags_to_encode:
            logger.info(f"Vectorizando {len(frags_to_encode):,} fragmentos relevantes por lotes...")
            embs_c = self.bi_model.encode(
                frags_to_encode,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for k, orig_idx in enumerate(frag_indices):
                combined = 0.65 * embs_t[orig_idx] + 0.35 * embs_c[k]
                embs_doc[orig_idx] = combined / np.linalg.norm(combined)

        # 4. Inferencia matricial de similitud coseno
        logger.info(f"Ejecutando producto matricial ({embs_doc.shape} x {self.matriz_centroides.shape})...")
        sims = np.dot(embs_doc, self.matriz_centroides.T)
        idx_max = np.argmax(sims, axis=1)
        scores = np.max(sims, axis=1)

        # 5. Aplicar umbrales adaptativos y filtrado anti-bleed
        data_process: list[dict[str, Any]] = []
        relevantes_count = 0
        ruido_count = 0

        for i in range(n_rows):
            cat = self.nombres_cats[idx_max[i]]
            score = float(scores[i])
            es_corta = es_corta_list[i]
            es_macro = es_macro_list[i]
            banco_prominente = banco_prominente_list[i]
            banco_en_titular = banco_en_titular_list[i]
            rol_analista = rol_analista_list[i]

            if es_corta:
                tau = 0.30 if (banco_en_titular or es_macro) else 0.38
            else:
                tau = 0.38 if (banco_prominente or es_macro) else 0.48

            if rol_analista:
                es_ruido = True
            elif not es_macro and not banco_prominente:
                es_ruido = True
            else:
                es_ruido = (cat.lower() == "ruido" or "ruido" in cat.lower()) or (score < tau)

            if es_ruido:
                ruido_count += 1
            else:
                relevantes_count += 1

            data_process.append(
                {
                    "ticker": tickers[i],
                    "date": dates[i],
                    "title": titulares[i],
                    "process_text": fragmentos[i] if fragmentos[i] else "(Titular directo sin cuerpo)",
                    "category": cat,
                    "confianza": round(score * 100, 2),
                    "is_ruido": es_ruido,
                }
            )

        elapsed = time.time() - t_start
        logger.info(
            f"Categorización completada en {elapsed:.1f} s (~{n_rows / max(elapsed, 0.001):.1f} registros/s). "
            f"Relevantes: {relevantes_count:,} ({relevantes_count / n_rows * 100:.1f}%) | "
            f"Ruido: {ruido_count:,} ({ruido_count / n_rows * 100:.1f}%)"
        )
        return data_process


# Alias de retrocompatibilidad
ModelCategorizar = CategorizadorArticulos
