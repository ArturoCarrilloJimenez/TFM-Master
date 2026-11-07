# 📋 Auditoría de Calidad y Origen de Falsos Positivos (FP) — Clasificación Semántica Zero-Shot (2010–2023)

**Dataset auditado (Capa Silver):** `processed/huggingface/multisource_bancos_2010_2023.parquet`  
**Universo:** US Big 6 (`BAC`, `C`, `GS`, `JPM`, `MS`, `WFC`)  
**Ventana temporal:** `2010–2023` (**89,846 registros totales**: `37,359` útiles [`is_ruido == False`, `41.58%`] y `52,487` descartados como ruido [`is_ruido == True`, `58.42%`]).

---

## 1. Matriz de Auditoría Cualitativa Muestral ($N = 60$ noticias)

A partir de un muestreo aleatorio estratificado (`random_state = 42`) de **5 noticias útiles (`is_ruido == False`)** y **5 noticias descartadas (`is_ruido == True`)** por cada uno de los 6 bancos (**30 útiles + 30 de ruido = 60 noticias**), se evaluó manualmente la calidad del filtrado tomando:
* **Clase Positiva:** Noticia que **SÍ sirve** (`is_ruido == False`, señal fundamental o bursátil del banco).
* **Clase Negativa:** **Ruido** (`is_ruido == True`, noticias sobre terceros, menciones tangenciales o resúmenes genéricos).

### 1.1. Resumen de la Matriz de Confusión Muestral ($N = 60$)

| Predicción del Clasificador \ Realidad Manual | Real: Útil (Señal Financiera del Banco) | Real: Ruido (Terceros / General / Sin Contexto) | Total Muestra | Métrica Muestral |
| :--- | :---: | :---: | :---: | :---: |
| **Predicho: SÍ Sirve (`is_ruido == False`)** | **TP = 18** | **FP = 12** | **30** (5 por banco) | **Precisión (PPV):** `60.0%` |
| **Predicho: Ruido (`is_ruido == True`)** | **FN = 5** | **TN = 25** | **30** (5 por banco) | **Valor Predictivo Negativo (NPV):** `83.3%` |
| **Total Real** | **23** | **37** | **60** | **Exactitud Global (Accuracy):** `71.7%` |

* **Sensibilidad (Recall sobre señal útil):** `78.3%` ($18 / 23$)
* **Especificidad (Tasa de rechazo de ruido real):** `67.6%` ($25 / 37$)

---

### 1.2. Evaluación Detallada de la Muestra Útil (`is_ruido == False`: 18 TP vs. 12 FP)

| # | Fecha | Ticker | Categoría | Confianza | Contexto (`process_text`) | Titular (`title`) | Veredicto | Justificación Técnica |
| :-: | :---: | :-: | :-: | :-: | :---: | :--- | :-: | :--- |
| **0** | 2010-07-19 | `BAC` | `macro` | 36.79% | Sin cuerpo | *Citi: Moody's Raises Outlook to Stable* | ❌ **FP** | Habla exclusivamente de Citigroup (`C`); sin cuerpo ni mención a `BAC`. |
| **1** | 2010-10-20 | `BAC` | `m&a` | 52.90% | Con cuerpo | *Hot Stocks: Regional Banks Slide* | ❌ **FP** | Habla de bancos regionales (Comerica y M&I); no menciona a `BAC`. |
| **2** | 2011-01-03 | `BAC` | `m&a` | 45.02% | Con cuerpo | *Bank of America, Boeing, Borders in focus* | ⚠️ **FP** | Lista genérica de apertura de futuros sin hecho fundamental de `BAC`. |
| **3** | 2015-04-15 | `BAC` | `m&a` | 47.85% | Sin cuerpo | *Fast Money Picks For April 15: Bank Of America, Exxon Mobil, Intel* | ⚠️ **FP** | Mención televisiva multi-ticker sin cuerpo ni contexto. |
| **4** | 2016-03-20 | `BAC` | `m&a` | 42.02% | Sin cuerpo | *Bank of America Erroneously Declared Jim Chanos' $18M Mortgage Paid in Full* | ✅ **TP** | Evento operativo/hipotecario directo de `BAC`. |
| **5** | 2017-05-12 | `C` | `legal` | 53.55% | Con cuerpo | *Is Bank of America Really the "Stingiest" Big Bank?* | ✅ **TP** | Comparativa directa del rendimiento de depósitos entre `BAC`, `JPM`, `WFC` y `C`. |
| **6** | 2017-10-12 | `C` | `earnings` | 45.23% | Sin cuerpo | *Citi Reports Book Value Per Share of $78.81, Repurchased 81M Common Shares* | ✅ **TP** | Reporte contable y recompra de acciones de `C`. |
| **7** | 2017-11-03 | `C` | `m&a` | 44.71% | Con cuerpo | *High Pointe Capital Management Llc Buys Progressive Corp...* | ❌ **FP** | Nota automática de cartera de un fondo (13F) donde `C` es una posición menor. |
| **8** | 2018-06-20 | `C` | `legal` | 50.58% | Sin cuerpo | *'The auditor of Citi, Credit Suisse and Deutsche Bank was tipped off...'* | ✅ **TP** | Evento regulatorio/auditoría que involucra directamente a `C`. |
| **9** | 2019-07-15 | `C` | `earnings` | 58.78% | Con cuerpo | *Citigroup Kicks Off Bank Earnings Season With a Strong Result* | ✅ **TP** | Resultados trimestrales directos de `C`. |
| **10** | 2010-02-05 | `GS` | `macro` | 37.82% | Sin cuerpo | *Markets Reserve Judgement on Mixed Jobs Numbers* | ❌ **FP** | Noticia macro genérica sin cuerpo ni mención a `GS`. |
| **11** | 2016-03-16 | `GS` | `m&a` | 53.83% | Con cuerpo | *Why Goldman is Set to Acquire Startup Firm Honest Dollar* | ✅ **TP** | Operación real de M&A ejecutada por `GS`. |
| **12** | 2020-07-13 | `GS` | `earnings` | 51.44% | Con cuerpo | *Earnings Season Kicks Off Amid Pandemic as Companies Pull Guidance* | ✅ **TP** | Previa directa del inicio de resultados trimestrales de `GS` y `JPM`. |
| **13** | 2021-04-06 | `GS` | `earnings` | 40.59% | Con cuerpo | *Goldman Sachs Group's Preferred Stock, Series K Shares Cross 5.5% Yield Mark* | ✅ **TP** | Cotización y rendimiento por dividendo de acciones preferentes de `GS`. |
| **14** | 2021-10-10 | `GS` | `macro` | 52.90% | Con cuerpo | *Goldman cuts forecast for U.S. economic growth in 2021 and 2022* | ⚠️ **FP** | Rol de `GS` como casa de análisis macroeconómico, no evento corporativo propio. |
| **15** | 2015-08-25 | `JPM` | `m&a` | 34.72% | Sin cuerpo | *J.P. Morgan Chase upgraded to buy from outperform at CLSA Americas* | ✅ **TP** | Mejora de recomendación bursátil directamente sobre la acción de `JPM`. |
| **16** | 2018-11-14 | `JPM` | `legal` | 48.69% | Sin cuerpo | *Banks sink after Waters warns that bank reg rollbacks will end* | ✅ **TP** | Impacto regulatorio directo sobre la gran banca estadounidense. |
| **17** | 2019-03-19 | `JPM` | `m&a` | 38.97% | Sin cuerpo | *JPMorgan Offers Checkless Accounts for Low-Income Customers* | ✅ **TP** | Lanzamiento comercial de producto bancario de `JPM`. |
| **18** | 2019-04-12 | `JPM` | `earnings` | 52.62% | Sin cuerpo | *J.P. Morgan Chase Q1 fixed income market revenue down 8%, equity markets...* | ✅ **TP** | Cifras directas de ingresos del 1T de `JPM`. |
| **19** | 2023-06-12 | `JPM` | `macro` | 59.49% | Con cuerpo | *Citigroup Warns Bond Traders Are Misreading Inflation Ahead of CPI* | ❌ **FP** | Titular de Citigroup con cuerpo tipo boletín de titulares concatenados. |
| **20** | 2011-01-28 | `MS` | `macro` | 45.47% | Sin cuerpo | *A much better bet on inflation than gold* | ❌ **FP** | Titular de opinión macro sin cuerpo ni mención a `MS`. |
| **21** | 2013-11-26 | `MS` | `legal` | 44.58% | Sin cuerpo | *Reuters Reports DOJ Investigating Morgan Stanley for Hiring Practices in China* | ✅ **TP** | Investigación del Departamento de Justicia (DOJ) sobre `MS`. |
| **22** | 2017-11-09 | `MS` | `earnings` | 39.86% | Con cuerpo | *Why Snap Inc Stock Fell Thursday* | ❌ **FP** | Rol de `MS` como analista emitiendo un *downgrade* tras los resultados de Snap Inc. |
| **23** | 2021-07-12 | `MS` | `earnings` | 45.03% | Con cuerpo | *1 Value Stock and 1 Growth Stock I'm Watching Now* | ✅ **TP** | Analiza la duplicación del dividendo de `MS` tras aprobar los *stress tests* de la Fed. |
| **24** | 2022-08-29 | `MS` | `legal` | 44.24% | Con cuerpo | *S.Korea inspecting stock short-selling, starting with Morgan Stanley* | ✅ **TP** | Inspección regulatoria directa a la filial de `MS` en Corea del Sur. |
| **25** | 2011-04-20 | `WFC` | `m&a` | 46.17% | Con cuerpo | *Financials rise but Wells Fargo misses the party* | ✅ **TP** | Caída del 3.5% de `WFC` tras presentar ingresos del 1T inferiores a lo esperado. |
| **26** | 2016-02-13 | `WFC` | `macro` | 39.61% | Con cuerpo | *Warren Buffett: 'I Am Not Worried' About Market* | ✅ **TP** | Desempeño de `WFC` (-13.3%) dentro de las principales posiciones de Berkshire Hathaway. |
| **27** | 2017-06-30 | `WFC` | `m&a` | 42.25% | Con cuerpo | *Citigroup (C) Scales 52-Week High on Investors' Optimism* | ✅ **TP** | Aprobación del plan de capital de `WFC`, `JPM` y `BAC` por la Reserva Federal. |
| **28** | 2017-12-21 | `WFC` | `m&a` | 36.18% | Sin cuerpo | *Buckingham Upgrades Wells Fargo to Buy* | ✅ **TP** | Upgrade de recomendación bursátil directamente sobre `WFC`. |
| **29** | 2022-04-18 | `WFC` | `macro` | 46.04% | Con cuerpo | *Stocks edge lower at the close, 10-year Treasury yield hits highest level...* | ❌ **FP** | Resumen genérico de cierre de mercado (viajes y semiconductores) sin mención a `WFC`. |

---

### 1.3. Evaluación Detallada de la Muestra Descartada (`is_ruido == True`: 25 TN vs. 5 FN)

* **✅ 25 Verdaderos Negativos (TN — `83.3%` de pureza en el descarte):**
  * **Descarte correcto del banco actuando como analista de terceros:** Filas **2** (`BAC`: *Credit Suisse opina sobre JPM*) y **19** (`JPM`: *el analista de JPM Stephen Tusa opina sobre General Electric*, descartado con `31.63%`).
  * **Descarte correcto de noticias de empresas ajenas sin relación con el banco:** Filas **3** (`BAC` - Apple), **13** (`GS` - Quit Genius / OMS), **16** (`JPM` - Kohl's), **17** (`JPM` - Apollo Group), **18** (`JPM` - artículo sobre BAC), **21** (`MS` - artículo sobre JPM), **24** (`MS` - dividendo de MarketAxess), **26** (`WFC` - dividendo de Itaú Unibanco), **27** (`WFC` - IPO de Vroom) y **28** (`WFC` - RedHat adquiere StackRox).
  * **Descarte correcto de alertas automáticas de ETFs y resúmenes genéricos:** Filas **0, 1, 4** (`BAC`), **7, 9** (`C`), **10, 11, 12** (`GS`), **15** (`JPM`), **20, 23** (`MS`), **25, 29** (`WFC`).

* **⚠️ 5 Falsos Negativos (FN — `16.7%` de pérdida por umbral de confianza `< 35%` o titulares compuestos):**
  1. **Fila 5 (`C`, 2010-04-26, `32.20%`):** *"Citi slips as Treasury announces sale"* (venta de 1.500 millones de acciones de Citi por el Tesoro de EE. UU.; cayó justo por debajo del umbral del `35%`).
  2. **Fila 6 (`C`, 2011-11-28, `29.93%`):** *"Citigroup shares add 5.1% in preopen"* (titular corto sin cuerpo con confianza `< 35%`).
  3. **Fila 8 (`C`, 2018-05-30, `33.84%`):** *"Italy Turmoil Puts These 6 Bank Stocks at Risk — Turmoil in Italy: Citigroup (C)"* (confianza `< 35%`).
  4. **Fila 14 (`GS`, 2022-11-16, `31.36%`):** *"Dell reaches $1 billion settlement over disputed 2018 stock swap"* (resuelve demandas contra `GS` como asesor de la operación; confianza `< 35%`).
  5. **Fila 22 (`MS`, 2015-07-20, `47.27%` en `ruido`):** *"Mid-Morning Market Update: Markets Mostly Flat; Morgan Stanley Beats Q2 Views"* (el encabezado *Mid-Morning Market Update* arrastró la predicción hacia `ruido` pese a que la segunda mitad del titular reportaba resultados de `MS`).

---

## 2. Cuantificación del Origen de los Falsos Positivos (FP) en el Dataset Completo ($N = 89,846$)

La auditoría cualitativa reveló que la mayoría de los **Falsos Positivos (`FP`)** provienen de dos patrones estructurales muy concretos:
1. **Foco FP #1 (Titulares huérfanos sin cuerpo):** Registros con `process_text == "(Titular directo sin cuerpo)"` donde ni el cuerpo original ni el titular mencionan realmente al banco asignado en la columna `ticker` del dataset bruto.
2. **Foco FP #2 (Rol de casa de análisis sobre terceros):** Noticias con cuerpo donde el titular habla de otra compañía (sin mencionar al banco) y el banco aparece únicamente en el cuerpo con jerga de analista (`upgrades`, `downgrades`, `price target`, `analyst`, `forecasts`, `top picks`, etc.).

Al medir la incidencia exacta de ambos fenómenos sobre los **89,846 registros** del periodo `2010–2023`, se obtienen los siguientes resultados:

### 2.1. Volumen de `(Titular directo sin cuerpo)` en el Dataset Completo (`N = 89,846`)

| Categoría de Registro | Volumen (Noticias) | % sobre el Total (`89,846`) | Destino en el Clasificador Actual |
| :--- | :---: | :---: | :--- |
| **Total con `(Titular directo sin cuerpo)`** | **23,010** | **25.61%** | `10,603` pasaron como útiles (`46.1%`) y `12,407` fueron a ruido (`53.9%`). |
| ├── **Con mención explícita del banco en el `title`** *(Titulares legítimos tipo flash)* | **9,211** | **10.25%** | `7,085` pasaron como útiles (`76.9%`) y `2,126` fueron a ruido (`23.1%`, por confianza `< 35%`). |
| └── **SIN mención del banco en el `title` ni cuerpo** *(Huérfanas totales / FP en potencia)* | **13,799** | **15.36%** | `10,281` se filtraron bien como ruido (`74.5%`), pero **`3,518` se colaron como útiles (`FP`)**. |

---

### 2.2. Impacto dentro de las Noticias que SÍ Sirven (`is_ruido == False`, `N = 37,359`)

Dentro de las **37,359 noticias útiles** retenidas en la capa Silver:

1. **Desglose de las 10,603 noticias sin cuerpo (`28.38%` de las útiles):**
   * ✅ **7,085 noticias (`18.96%` de las útiles):** Son titulares directos sin cuerpo que **SÍ mencionan explícitamente al banco** en el titular (señal válida de titulares financieros tipo Reuters/Benzinga).
   * ❌ **3,518 noticias (`9.42%` de las útiles — Foco FP #1):** Son **Falsos Positivos huérfanos puros**; no tienen cuerpo extraído y **tampoco mencionan al banco ni a sus alias en el titular**.
2. **Desglose de las 26,756 noticias con cuerpo (`71.62%` de las útiles):**
   * ⚠️ **2,841 noticias (`7.60%` de las útiles — Foco FP #2):** El titular **no menciona al banco** y el fragmento extraído contiene patrones léxicos de recomendación o análisis sobre terceros.

> 📌 **Síntesis de Magnitud Global:** Entre ambos focos (`3,518` titulares huérfanos sin banco + `2,841` notas de analista sobre terceros) suman **6,359 registros**, equivalentes al **17.02% de las noticias útiles (`37,359`)**. Su depuración deja un núcleo de **31,000 noticias de alta pureza (`83.0%`)**.

---

### 2.3. Distribución por Entidad Bancaria dentro de las Noticias Útiles (`is_ruido == False`)

El comportamiento presenta una asimetría clara por entidad:
* **`JPM`, `MS` y `BAC`** concentran el **82.1%** de todos los titulares huérfanos sin cuerpo (`FP #1`).
* **`C` y `GS`** apenas presentan titulares huérfanos (`0.0%` y `2.5%`), pero registran la mayor proporción relativa de menciones en rol de casa de análisis (`12.3%` y `8.5%`).

| Banco (`ticker`) | Total Útiles (`is_ruido=False`) | Total `(Titular sin cuerpo)` | % Sin Cuerpo | ✅ Sin Cuerpo pero **CON** Banco en `title` (Válidas) | ❌ **FP #1:** Sin Cuerpo y **SIN** Banco en `title` (Huérfanas) | % FP #1 (Huérfanas) | ⚠️ **FP #2:** Con Cuerpo, Sin Banco en `title` + Jerga Analista | % FP #2 (Analista) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **`BAC`** | 4,295 | 2,006 | 46.7% | 1,300 | **706** | **16.4%** | 248 | 5.8% |
| **`C`** | 5,098 | 607 | 11.9% | 607 | **0** | **0.0%** | 629 | **12.3%** |
| **`GS`** | 6,708 | 455 | 6.8% | 285 | **170** | **2.5%** | 573 | **8.5%** |
| **`JPM`** | 6,710 | 2,720 | 40.5% | 1,615 | **1,105** | **16.5%** | 408 | 6.1% |
| **`MS`** | 7,084 | 3,208 | 45.3% | 2,131 | **1,077** | **15.2%** | 433 | 6.1% |
| **`WFC`** | 7,464 | 1,607 | 21.5% | 1,147 | **460** | **6.2%** | 550 | 7.4% |
| **TOTAL** | **37,359** | **10,603** | **28.4%** | **7,085** | **3,518** | **9.4%** | **2,841** | **7.6%** |

---

## 3. Diagnóstico de las "Puertas Traseras" del Clasificador Actual y Estrategias para Reducir el % de FP

Al cruzar los **12 Falsos Positivos (`FP`)** de la muestra con la lógica actual del clasificador, se identifican **5 causas exactas (puertas traseras lógicas)** por las que se cuela el ruido. A continuación se detalla cada causa, qué filas de la auditoría explica, qué volumen representa en el dataset y cómo pulirla conceptualmente:

---

### Estrategia 1: Cerrar la "Puerta Trasera de Rescate" en Noticias Sin Cuerpo y Sin Banco (`Elimina 3,518 FP = -9.42% directo`)
* **¿Qué ocurre hoy?**  
  Actualmente existe una regla de excepción pensada para rescatar titulares cortos de un solo ticker (`ticker_count == 1`): si una noticia no tiene cuerpo y no menciona al banco ni en el titular ni en el texto, aun así se aprueba como útil si supera el `35%` de confianza en una categoría financiera o si cae en `macro` (`> 30%`).
* **¿Qué FP provocó en la muestra?**  
  * **Fila 0 (`BAC`):** *"Citi: Moody's Raises Outlook to Stable"* (el dataset original la etiquetó erróneamente en `BAC`; como no tenía cuerpo y hablaba de un rating bancario, pasó limpia para `BAC` cuando era de `Citigroup`).
  * **Fila 10 (`GS`):** *"Markets Reserve Judgement on Mixed Jobs Numbers"* (titular macro sin cuerpo ni mención a `GS`).
  * **Fila 20 (`MS`):** *"A much better bet on inflation than gold"* (titular de opinión sobre inflación/oro sin cuerpo ni mención a `MS`).
* **¿Cómo pulirlo?**  
  **Regla de Anclaje Obligatorio en Titulares Huérfanos:** Si una noticia no tiene cuerpo (`(Titular directo sin cuerpo)`), **exigir sin excepciones que el titular mencione al banco (o a alguno de sus alias)**, o bien que el titular contenga un término explícito de banca sistémica (`Big Banks`, `Wall Street Banks`, `US Banks`, `Stress Test`). Si no menciona ni al banco ni a la banca en su conjunto, es ruido automático.

---

### Estrategia 2: Eliminar el "Salvoconducto Macro" cuando el Cuerpo NO Menciona al Banco
* **¿Qué ocurre hoy?**  
  Cuando un artículo sí tiene cuerpo pero el buscador de fragmentos **no encuentra al banco en ningún párrafo**, el sistema hace un *fallback* tomando los primeros 400 caracteres del artículo. Si en esos 400 caracteres o en el titular aparece cualquier palabra general como `economy`, `treasury`, `inflation` o `financial sector`, el clasificador activa la bandera `es_macro` y **deja pasar la noticia aunque el banco del `ticker` no aparezca en todo el artículo**.
* **¿Qué FP provocó en la muestra?**  
  * **Fila 1 (`BAC`):** *"Hot Stocks: Regional Banks Slide"* (el cuerpo habla de Comerica y Marshall & Ilsley; `BAC` no aparece, pero como dice *"financial sector"* y *"economy"*, se activó el salvoconducto macro).
  * **Fila 2 (`BAC`):** *"Bank of America, Boeing, Borders in focus"* (el cuerpo solo habla de futuros de EE. UU. y *"U.S. growth"* sin mencionar a `BAC`).
  * **Fila 29 (`WFC`):** *"Stocks edge lower at the close, 10-year Treasury yield hits highest level since 2018"* (el cuerpo habla de viajes y semiconductores; `WFC` no aparece jamás, pero la palabra *"Treasury"* en el titular le dio pase libre).
* **¿Cómo pulirlo?**  
  **Condicionar la categoría `Macro` a presencia real o impacto bancario explícito:** Una noticia con cuerpo donde el banco **no es mencionado en todo el texto** no puede aprobarse solo por contener la palabra `Treasury`, `economy` o `inflation`. Para aprobar una noticia macro sin mención directa del banco en el cuerpo, el titular o el fragmento debe mencionar explícitamente al **sector bancario o a la Reserva Federal regulando bancos** (ej. `bank stocks`, `banking sector`, `Fed stress test`, `capital requirements`), no simples cierres generales del S&P 500 o del bono a 10 años.

---

### Estrategia 3: Regla de "Exclusión por Protagonismo de Otro Banco Competidor" (*Cross-Bank Bleed*)
* **¿Qué ocurre hoy?**  
  A veces el titular nombra explícitamente a **un único banco competidor** (por ejemplo, *"Citi: ..."* o *"Citigroup Warns..."*), pero la fila pertenece a `BAC` o `JPM` porque en algún lugar del artículo original (o en un boletín de titulares concatenados) aparecía el nombre del otro banco.
* **¿Qué FP provocó en la muestra?**  
  * **Fila 0 (`BAC`):** *"Citi: Moody's Raises Outlook to Stable"*.
  * **Fila 19 (`JPM`):** *"Citigroup Warns Bond Traders Are Misreading Inflation Ahead of CPI"* (cuerpo compuesto por titulares pegados sin separación).
* **¿Cómo pulirlo?**  
  **Filtro de Sujeto Competidor en Titular:** Si el titular menciona explícitamente a **uno solo** de los otros 5 bancos del *US Big 6* (ej. menciona a `Citigroup` pero **no** menciona a `JPM` ni palabras colectivas como `Big Banks`, `peers`, `rivals`, `along with`), y en el fragmento extraído el banco objetivo (`JPM`) no tiene ningún verbo de acción corporativa propia, debe descartarse como contaminación cruzada (*ticker bleed*).  
  *(Nota: Esto respeta casos legítimos como la **Fila 5** —"Is Bank of America Really the Stingiest Big Bank?"— o la **Fila 27** —"Citigroup Scales 52-Week High... Other 33 big banks including JPM, BAC and WFC received approval"— porque ahí el fragmento sí compara explícitamente métricas o aprobaciones de capital de los demás bancos).*

---

### Estrategia 4: Ampliar la Detección de "Banco como Casa de Análisis / Pronosticador" (`Mitiga hasta 2,841 FP = -7.60%`)
* **¿Qué ocurre hoy?**  
  El detector actual de rol de analista es muy rígido: solo salta si encuentra exactamente las palabras `"{banco} analyst"`, `"analyst at {banco}"`, `"{banco}'s analyst"` o `"{banco} reiterated"`.
* **¿Qué FP provocó en la muestra?**  
  * **Fila 22 (`MS`):** *"Why Snap Inc Stock Fell Thursday — ...an analyst downgrade **from** Morgan Stanley following the earnings release"*. Como decía *"downgrade from Morgan Stanley"* en vez de *"analyst at Morgan Stanley"*, burló la regla y se clasificó como `earnings` de `MS` cuando los `earnings` eran de **Snap Inc**.
  * **Fila 14 (`GS`):** *"Goldman cuts forecast for U.S. economic growth in 2021 and 2022"*. Aquí Goldman no sufre un evento financiero propio, sino que publica un informe macroeconómico.
* **¿Cómo pulirlo?**  
  1. **Ampliar las construcciones sintácticas de recomendación sobre terceros:** Cuando el titular **no menciona al banco** (habla de otra empresa como *Snap*, *Apple*, *Tesla*), marcar como rol de analista cualquier aparición en el cuerpo de patrones de emisión de juicio: `downgrade from {banco}`, `upgrade from {banco}`, `downgraded by {banco}`, `upgraded by {banco}`, `{banco} lowered its price target`, `{banco} raised its price target`, `{banco} initiates coverage`, `according to {banco}`.
  2. **Separar "Pronósticos de Casa de Estudios" de "Hechos del Banco":** Cuando el titular tiene la estructura `{Banco} cuts/raises forecast for [GDP / oil / S&P 500 / inflation]` o `{Banco} warns traders...`, el banco es el **emisor de una opinión macro**, no el sujeto pasivo de un evento financiero.

---

### Estrategia 5: Filtrar Notas Robotizadas de Fondos (13F) y Listas Multi-Sector de TV/Apertura
* **¿Qué ocurre hoy?**  
  Existen artículos generados por plantillas automáticas que listan compras de fondos pequeños (*13F filings*) o listas de tickers variados de distintos sectores (*"Fast Money Picks: Bank of America, Exxon Mobil, Intel"*). Como los centroides de `m&a` contienen términos como `buys`, `acquires` o `stake`, el modelo confunde la compra de 20.000 acciones por parte de un fondo pequeño con una operación de M&A del banco.
* **¿Qué FP provocó en la muestra?**  
  * **Fila 7 (`C`):** *"High Pointe Capital Management Llc Buys Progressive Corp, Micro Focus... Citigroup Inc (C) - 27,520 shares, 2.93% of the total portfolio."* (clasificada erróneamente como `m&a` con `44.71%`).
  * **Fila 2 (`BAC`):** *"Bank of America, Boeing, Borders in focus"*.
  * **Fila 3 (`BAC`):** *"Fast Money Picks For April 15: Bank Of America, Exxon Mobil, Intel"*.
* **¿Cómo pulirlo?**  
  1. **Filtro de Plantillas 13F / Portafolios:** Descartar como `ruido` los textos con patrón de declaración patrimonial de fondos (`"% of the total portfolio"`, `"shares, X.XX% of"`, `"holdings in"`, `"buys X,XXX shares"` cuando el comprador en el titular es un `Capital Management / Advisors / LLC` ajeno al banco).
  2. **Filtro de Listas Multi-Ticker sin Cuerpo:** Si un titular tipo lista (`"Fast Money Picks"`, `"Stocks in focus"`, `"Stocks to Watch"`) enumera 3 o más compañías de sectores distintos y **carece de cuerpo específico sobre el banco**, marcarlo como ruido.

---

## 4. Resumen del Impacto Esperado al Aplicar estas 5 Mejoras

| Nivel de Pulido | Foco Atacado | Noticias FP Estimadas que se Eliminan | Nuevo Volumen Útil Estimado | Mejora en Precisión Muestral (sobre los 30 útiles de la muestra) |
| :--- | :--- | :---: | :---: | :---: |
| **Estado Actual** | Sin post-filtrado adicional | `0` | `37,359` (100%) | **60.0%** (`18 TP / 30`) |
| **Nivel 1 (Estrategias 1 + 2)** | Titulares huérfanos sin banco + Salvoconducto Macro sin banco en cuerpo | **~4,200 – 4,800** | `~32,500` | Sube a **~78.3%** (elimina los FP de las filas `0, 1, 2, 10, 20, 29`) |
| **Nivel 2 (+ Estrategia 3 y 4)** | Contaminación de otro banco en titular + Nuevos patrones de Analista (`downgrade from`, `cuts forecast`) | **~2,000 – 2,500** adicionales | `~30,200` | Sube a **~90.0%** (elimina además los FP de las filas `14, 19, 22`) |
| **Nivel 3 (+ Estrategia 5)** | Plantillas robotizadas 13F (`% of the total portfolio`) + Listas `Fast Money Picks` | **~500 – 800** adicionales | **`~29,500`** | Sube a **>94% – 100%** en la muestra (elimina también las filas `3 y 7`) |


---

## 5. Resultados Empíricos y Conclusiones del Post-Procesamiento (`zero_shot_clasiffication_FP_Analisis.ipynb`)

Tras implementar y ejecutar la auditoría exhaustiva por cubetas en `src/notebooks/zero_shot_clasiffication_FP_Analisis.ipynb` sobre las **37,359 noticias útiles (`is_ruido == False`)** publicadas en `tfm-master-prediccion-bancos-usa/processed/huggingface/df_util/df_util_2010_2023.parquet`, se obtuvieron los siguientes resultados empíricos:

### 5.1. Radiografía de Anclaje de la Entidad Bancaria (`N = 37,359`)

| Tipo de Anclaje de la Entidad en el Texto | Noticias | % sobre Total Útiles (`37,359`) | Diagnóstico |
| :--- | :---: | :---: | :--- |
| **1. Anclaje Doble (Banco en Titular y en Cuerpo)** | **7,756** | **20.76%** | Señal de máxima pureza corporativa. |
| **2. Titular Flash Válido (Banco en Titular, Sin Cuerpo)** | **7,085** | **18.96%** | Titulares directos válidos (ej. Reuters / Benzinga). |
| **3. Banco en Titular, Cuerpo por Fallback** | **848** | **2.27%** | El titular nombra al banco, aunque el extracto tomó el inicio del texto. |
| **4. Solo en Cuerpo (Titular NO menciona al banco)** | **14,217** | **38.06%** | Conviven noticias multipárrafo válidas con menciones en rol de analista y plantillas 13F. |
| **5. Huérfano Sin Cuerpo (NI en Titular NI Cuerpo) `[FP]`** | **3,518** | **9.42%** | Anomalía sin cuerpo donde el banco no es nombrado (`2,165` sin mención al sector bancario). |
| **6. Fallback Sin Banco (Con Cuerpo, pero NI en Titular NI Cuerpo) `[FP]`** | **3,935** | **10.53%** | Anomalía con cuerpo donde el extractor no halló al banco en el artículo (`2,030` sin mención al sector bancario). |

---

### 5.2. Cuantificación de las 5 Cubetas de Falsos Positivos vs. Núcleo Limpio (`TP`)

La aplicación jerárquica de las 5 reglas identificó **6,225 Falsos Positivos (`16.66%`)** y consolidó un **Núcleo Limpio (`TP`) de 31,134 noticias (`83.34%`)**:

| Cubeta de Auditoría (`fp_bucket`) | Noticias | % sobre Total Útiles (`37,359`) | Evaluación tras Inspección Muestral (Bloque 4) |
| :--- | :---: | :---: | :--- |
| ✅ **`TP: Núcleo Limpio (Señal Válida)`** | **31,134** | **83.34%** | Núcleo depurado con cobertura completa en los 6 bancos (2010–2023). |
| ❌ **`FP_1: Huérfana Sin Cuerpo ni Banco`** | **2,165** | **5.80%** | **100% Ruido puro:** Titulares sin cuerpo de otros bancos o temas ajenos. |
| ❌ **`FP_2: Fallback Con Cuerpo pero Sin Banco`** | **2,030** | **5.43%** | **100% Ruido puro:** Resúmenes generales de mercado donde el banco nunca es mencionado. |
| ⚠️ **`FP_4: Banco como Analista / Pronóstico Macro`** | **980** | **2.62%** | Captura rol de analista, pero el término `estimates` en el regex atrapó también comparativas de *Zacks Consensus Estimate* sobre resultados reales del banco (refinable). |
| ⚠️ **`FP_3: Otro Banco en Titular (Cross-Bleed)`** | **786** | **2.10%** | Captura noticias cuyo titular protagoniza un competidor sin términos comparativos en el extracto. |
| ❌ **`FP_5: Plantilla 13F / ETF / Lista TV`** | **264** | **0.71%** | **100% Ruido robotizado:** Notas automáticas de fondos (`% of the total portfolio`), alertas de ETFs y listas de TV (`Fast Money Picks`). |

#### Desglose por Banco (`Conteo por Cubeta y % Total de FP`)

| Banco (`ticker`) | `FP_1` (Huérfana) | `FP_2` (Fallback) | `FP_3` (Cross-Bleed) | `FP_4` (Analista) | `FP_5` (13F / TV) | ✅ **`TP: Núcleo Limpio`** | Total Antes | `% FP Detectado` |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **`BAC`** | 367 | 395 | 57 | 25 | 57 | **3,394** | 4,295 | **21.0%** |
| **`C`** | 0 | 43 | 233 | 277 | 83 | **4,462** | 5,098 | **12.5%** |
| **`GS`** | 133 | 443 | 148 | 207 | 28 | **5,749** | 6,708 | **14.3%** |
| **`JPM`** | 738 | 691 | 62 | 38 | 2 | **5,179** | 6,710 | **22.8%** |
| **`MS`** | 663 | 154 | 157 | 191 | 17 | **5,902** | 7,084 | **16.7%** |
| **`WFC`** | 264 | 304 | 129 | 242 | 77 | **6,448** | 7,464 | **13.6%** |
| **TOTAL** | **2,165** | **2,030** | **786** | **980** | **264** | **31,134** | **37,359** | **16.7%** |

---

### 5.3. Impacto en la Cobertura Temporal (2010–2023) Post-Limpieza

Al verificar el mapa de calor anual (`Banco × Año`) después de retirar los `6,225` Falsos Positivos (Bloque 5):
* **Cobertura anual intacta (`100%`):** Ningún banco queda con `0` noticias en ningún año entre 2010 y 2023 (`Años_con_0_Noticias = 0` para los 6 tickers).
* **Mínimos anuales tras la depuración:**
  * `BAC`: mínimo de **21 noticias** (en `2021`).
  * `C`: mínimo de **26 noticias** (en `2021`).
  * `GS`: mínimo de **81 noticias** (en `2015`).
  * `JPM`: mínimo de **13 noticias** (en `2022`).
  * `MS`: mínimo de **314 noticias** (en `2019`).
  * `WFC`: mínimo de **61 noticias** (en `2010`).

---

### 5.4. Validación Muestral Final (`N = 30`, Bloque 6) y Ajustes Finos Identificados

Al extraer una nueva muestra aleatoria estratificada de **5 noticias por banco (`N = 30`)** sobre el `TP: Núcleo Limpio`:
1. **Salto de Precisión Muestral:** La precisión subió del **60.0% original (`18 TP / 30`) al 83.3% (`25 TP / 30`)**, alcanzando un **100% de precisión (`15 TP / 15`) en `JPM`, `MS` y `WFC`**.
2. **Identificación de los 2 últimos ajustes para alcanzar `>95%` de pureza:**
   * **Ajuste A (Eliminar la excepción `sector_bancario_en_title` en `FP_2`):** 4 de los 5 FP residuales de la muestra final (Filas `2, 3, 11 y 14` en `BAC` y `GS`) eran artículos **con cuerpo** donde el banco no aparecía ni en el titular ni en el fragmento, pero que se habían salvado porque el titular contenía la palabra genérica `"banks"`, `"financial stocks"` o `"stress tests"`. Exigir que toda noticia con cuerpo mencione al banco en el titular o en el fragmento elimina el 100% de estos *fallbacks* vacíos (`1,905` registros adicionales).
   * **Ajuste B (Refinar el patrón léxico de `FP_4`):**
     * Retirar la palabra `"estimates"` de `PATRON_ANALISTA_AMPLIADO` para no descartar artículos de resultados trimestrales que citan el *"Zacks Consensus Estimate"*.
     * Ampliar `PATRON_PRONOSTICO_CASA` para admitir palabras intermedias entre el verbo y el sustantivo (ej. *"Goldman Sachs **lowers recession forecast**"* en la Fila `13`).
