# Memoria del TFM en LaTeX (`docs/memoria/`)

Estructura modular de la memoria del Trabajo Fin de Máster (**máximo 60 páginas en PDF** según normativa EBIS).

## 📂 Estructura de archivos

```text
docs/memoria/
├── main.tex                         # Documento maestro (portada, estilos, índices, bibliografía)
├── referencias.bib                  # Base bibliográfica BibLaTeX
├── figures/                         # Imágenes y gráficas exportadas (300 DPI / PDF / PNG)
└── chapters/
    ├── 01_introduccion.tex          # Capítulo 1: Contexto, problema, OE1-OE5 y alcance
    ├── 02_estado_del_arte.tex       # Capítulo 2: Literatura financiera, NLP y XAI
    ├── 03_arquitectura_ingesta.tex  # Capítulo 3: Always Free R2 Medallion y fuentes Bronze
    ├── 04_eda_preprocesamiento.tex  # Capítulo 4: EDA, Gatekeeper Few-Shot, Sentiment y PiT
    ├── 05_modelado.tex              # Capítulo 5: Feature Engineering, Boruta y Quantile Regression
    ├── 06_resultados.tex            # Capítulo 6: Métricas, ablación, TreeSHAP y backtesting
    ├── 07_despliegue.tex            # Capítulo 7: Airflow, Lambda, QA (tests/Sonar) y Streamlit
    ├── 08_conclusiones.tex          # Capítulo 8: Conclusiones, análisis ROI y líneas futuras
    └── anexos.tex                   # Anexos A y B: ADRs, esquemas y reproducción con uv
```

## 🛠️ Cómo compilar en local (Windows + MiKTeX)

Desde la raíz del repositorio en PowerShell:

```powershell
.\docs\memoria\compilar.ps1
```

Esto ejecuta automáticamente `xelatex -> biber -> xelatex -> xelatex` y genera `docs/memoria/main.pdf`.

## ☁️ Cómo compilar en Overleaf (Alternativa web)

1. Comprimir el contenido de `docs/memoria/` en un `.zip` (o sincronizar vía GitHub si disponéis de integración).
2. En Overleaf, abrir **Menu → Compiler** y seleccionar **`XeLaTeX`** (obligatorio por `fontspec` + `Noto Sans`).
3. Añadir cualquier gráfica nueva dentro de `figures/` e incluirla en el capítulo correspondiente con `\includegraphics[width=0.9\textwidth]{nombre_archivo.png}`.
4. Al finalizar, verificar que el PDF total no supere las **60 páginas** exigidas por EBIS.
