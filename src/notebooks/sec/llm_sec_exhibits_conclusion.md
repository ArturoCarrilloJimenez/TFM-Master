  El mejor resultado es claude-haiku-5-5 con el prompt sec_v2_rules: 94% de acierto en los 100 anexos, con 0 inversiones       
  (positivo predicho como negativo, o al revés) y coste de céntimos. La mayor parte del error restante apunta a etiquetas      
  dudosas del Ground Truth más que a fallos del modelo.                                                                        
                                                                                                                               
  Para sacar estas cifras he recalculado las métricas desde scratch/llm_sec_exhibits_pilot_cache.csv, sin hacer llamadas       
  nuevas.                                                                                                                      
                                                                                                                               
  Aciertos                                                                                                                     
                                                                                                                               
  Ranking sobre N = 100 (ctx_1200, batch_size 10):                                                                             
                                                                                                                               
   Modelo / prompt           │ Acc         │ F1          │ Rec+1       │ Rec0        │ Rec-1       │ Inv         │ Tok in+out
  ───────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────
   claude-haiku-5-5 v2_rules │ 0.94        │ 0.941       │ 0.95        │ 0.88        │ 1.00        │ 0           │ 662+178
   claude-haiku-5-5 v1       │ 0.92        │ 0.925       │ 0.92        │ 0.88        │ 0.96        │ 0           │ 621+164
   grok-4.3 v1               │ 0.92        │ 0.922       │ 0.92        │ 0.91        │ 0.93        │ 1           │ 393+62
   claude-sonnet-4-6 v1      │ 0.92        │ 0.921       │ 0.92        │ 0.88        │ 0.96        │ 1           │ 434+94
   gpt-4o-mini v1            │ 0.90        │ 0.899       │ 0.95        │ 0.82        │ 0.93        │ 1           │ 378+49
   gemini-2.5-flash v1       │ 0.86        │ 0.863       │ 0.87        │ 0.82        │ 0.89        │ 0           │ 412+71
   gpt-5-nano v1             │ 0.82        │ 0.809       │ 0.97        │ 0.79        │ 0.63        │ 6           │ 378+151
                                                                                                                               
  • Mejor que en prensa: el LLM sobre anexos llega a 92–94%, frente al 89% de prensa. Era previsible: los anexos tienen emisor 
  único y cifras explícitas.                                                                                                   
  • El Recall direccional es casi perfecto: con el mejor modelo, los 27 negativos del GT se detectan todos y 37 de 39 positivos.
  La matriz de confusión no tiene ningún error de signo opuesto.                                                               
  • Por Ítem 8-K: Item 8.01 (27 anexos), Item 5.02 (16) e Item 1.01 (2) salen al 100%. Item 2.02 (46) está en 89% y Item 7.01  
  en 88%.                                                                                                                      
  • Por banco: BAC, GS y MS al 100%, JPM 95%, WFC y C 88%.                                                                     
  • Las ablaciones no mejoran lo que ya hay. Se hicieron con claude-haiku-5-5 sobre una submuestra de 60 anexos:               
      • Subir el contexto a 2.000 caracteres da 0.883, sin mejora real frente a 1.200. Los 800 caracteres dan 0.867.           
      • batch_size 5, 10 o 15 da lo mismo. Con lotes de 15 el coste por anexo baja un poco, así que batch_size 10 o 15 es      
      suficiente.                                                                                                              
  • Coste de producción: para los ~974 anexos sustantivos, Haiku 5.5 sale a unos $0.15 en API estándar y $0.08 en Batch API.   
  grok-4.3 rondaría los $0.63. El tamaño de la factura no es el problema.                                                      
                                                                                                                               
  Errores                                                                                                                      
                                                                                                                               
  1. Inconsistencia del GT: con el mejor modelo, 6 de los 100 anexos fallan, y los diez modelos fallan sistemáticamente en los 
  mismos casos.                                                                                                                
      • Los fallos se concentran en trimestres de bancos cuyo Ground Truth puse como 0 por señales mixtas. Ejemplos: Citi Q3   
      2010, Citi Q4 2012 y WFC Q1/Q4 2016.                                                                                     
      • Los LLM leen esos trimestres como beneficio neto que sube o baja. Tienen un argumento razonable, y la regla "mixto = 0"
      que apliqué es discutible.                                                                                               
      • Hay casos donde el consenso va en contra de mi etiqueta y probablemente tiene razón. Por ejemplo, Citi Q2 2020         
      (beneficio $1.3B frente a $4.8B), JPM Q4 2013 y MS Q4 2011 los pusimos como -1 y la mayoría de modelos no lo ven así. Eso
      apunta a ambigüedad en el GT, no solo a error del modelo.                                                                
  2. Neutrales que se leen como direccionales: los 6 fallos del mejor modelo son 2 casos de 0 predicho como +1, 2 casos de 0   
  predicho como -1 y 2 casos de +1 predicho como 0. Entre ellos, la diapositiva del Investor Day de JPM 2017 (GT +1, modelo 0).
  3. Los modelos más baratos o antiguos pierden fiabilidad direccional: gpt-5-nano tiene 6 inversiones y 63% de Recall en      
  negativos, así que no sirve. gpt-4.1-mini empeora con el prompt v2.                                                          
  4. gemini-3.1-flash-lite y gemini-3.5-flash-lite no tienen resultados: el 503 de Google cortó esas ejecuciones, así que no   
  hay datos suyos. Habría que relanzarlos.                                                                                     
                                                                                                                               
  Cautelas                                                                                                                     
                                                                                                                               
  • Margen estadístico: con N = 100, un punto de acierto es un anexo. La diferencia entre v1 (0.92) y v2 (0.94) son 2 anexos y 
  no es significativa. Tampoco hay diferencia real entre los cuatro primeros modelos.                                          
  • Fuga de información: escribí las reglas del prompt v2 (planes de acciones, amortizaciones a la par, multas) a partir de    
  patrones que vi en estos mismos 100 casos. Es probable que 0.94 sobreestime el resultado en datos nuevos.                    
  • Ground Truth auditado por mí: parte de las etiquetas las puse yo leyendo el anexo. No es doble ciego, así que el techo real
  de acierto puede ser menor.                                                                                                  
                                                                                                                               
  Margen de mejora                                                                                                             
                                                                                                                               
  1. Auditar el GT antes de seguir afinando. Revisar a mano los ~14 anexos donde la mayoría de modelos discrepa del GT y fijar 
  una regla clara para los trimestres mixtos. Es lo que más va a cambiar las métricas.                                         
  2. Ampliar a ~250 anexos con un conjunto de validación separado. Hoy el 89% son EX-99, y solo hay 2 EX-2 y 9 EX-10, que son  
  fáciles. Con 150 anexos nuevos se mide sin fuga de información.                                                              
  3. Verificar la capa determinista del embudo. Los 4.557 anexos podados a $0 se tratan como neutros sin comprobar. Un muestreo
  de 100 (EX-4, EX-5, EX-99.2 deduplicados) confirmaría que no escondemos señal.                                               
  4. Reglas distintas para Item 2.02 y Item 7.01: ahí está el error restante. Un prompt de resultados separado, con una regla  
  explícita para trimestres mixtos, sería el siguiente experimento.                                                            
  5. Modelo de segunda opinión barato: Haiku 5.5 clasifica todo y solo los casos de baja confianza (p_neu entre 0.3 y 0.6)     
  pasan a claude-sonnet-4-6 o grok-4.3. Costaría céntimos extra y reduciría errores en Item 2.02.                              
  6. Completar gemini-3.1-flash-lite y gemini-3.5-flash-lite cuando Google deje de dar 503. Podrían ser una opción casi        
  gratuita en el Free Tier. 