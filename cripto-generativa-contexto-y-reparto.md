# Simulación condicional y stress testing de carteras cripto

**Documento de arranque — equipo de 3 personas**  
Fecha límite de entrega: 4 de septiembre de 2026

Este documento conserva el alcance y los criterios acordados al inicio. Sus
frases en futuro describen objetivos, no evidencia de ejecución; el estado y
los resultados comprobables se documentan en el `README.md` y en los notebooks
ejecutados.

---

## 1. El problema, en una frase

Queremos un modelo que, partiendo del estado actual del mercado de Bitcoin y Ethereum, **genere miles de trayectorias conjuntas plausibles para los próximos 30 días**, y utilizar esas trayectorias para medir cuánto puede perder una cartera de criptoactivos.

El proyecto no pretende predecir cuál será el precio de BTC o ETH dentro de un mes. Pretende estimar una **distribución de escenarios posibles**, condicionada por cómo se encuentra el mercado hoy.

---

## 2. Por qué esto no es trivial

La forma clásica de hacer stress testing en una cartera cripto es aplicar shocks prefijados:

- “Bitcoin cae un 30%”.
- “Ethereum cae un 40%”.
- “La correlación entre ambos sube hasta 0,90”.
- “Se repite una crisis parecida a marzo de 2020 o al colapso de FTX”.

Son escenarios comprensibles, pero arbitrarios. No indican si una caída conjunta determinada es un suceso relativamente frecuente o un episodio extremadamente improbable.

La alternativa es utilizar directamente el histórico: seleccionar movimientos pasados o hacer *bootstrap* de los retornos observados. Es un rival serio, pero tiene un límite. Bitcoin y, especialmente, Ethereum y las demás criptomonedas han atravesado pocos ciclos de mercado verdaderamente independientes. Disponer de miles de observaciones diarias u horarias no significa haber observado miles de crisis, cambios regulatorios o regímenes de liquidez diferentes.

**Lo que puede aportar un modelo generativo:** aprender la distribución conjunta de retornos, volatilidades y dependencias entre activos, y producir numerosas trayectorias coherentes con lo observado.

**Lo que no puede aportar:** crear información histórica que nunca existió. Si un régimen o una crisis no aparece en los datos de entrenamiento, no hay fundamento para afirmar que el modelo aprenderá a reproducirlo correctamente.

Generar 100.000 trayectorias reduce el error de Monte Carlo **condicionado al modelo**, pero no elimina el error de especificación ni convierte diez años de mercado en 100.000 historias independientes.

---

## 3. Alcance inicial

### Universo principal

El núcleo del proyecto será:

- `BTC-USD`
- `ETH-USD`

Se utilizarán pares frente a la misma divisa y, preferentemente, procedentes de un único mercado o de una fuente con una metodología de agregación estable.

### Extensión opcional

`SOL-USD` se incorporará únicamente si:

1. el modelo BTC–ETH está terminado;
2. el histórico disponible supera los controles de calidad;
3. queda tiempo para analizar por separado el problema de su menor historial.

No se incluirán inicialmente monedas estables, porque su dinámica económica y estadística es distinta. Tampoco se escogerán activos solo porque actualmente tengan una elevada capitalización: seleccionar exclusivamente los supervivientes de hoy introduciría sesgo de supervivencia.

### Frecuencia y horizonte

- Frecuencia preferida: **velas de 6 horas**.
- Horizonte generado: **30 días**.
- Longitud de cada trayectoria: **120 pasos**.
- Mercado continuo: calendario 24/7 y timestamps en UTC.

La frecuencia de seis horas ofrece un compromiso razonable: permite estudiar el camino seguido por los precios sin llevar el problema a una dimensionalidad excesiva. Si la fuente elegida no proporciona velas de seis horas de forma fiable, se descargarán velas horarias y se agregarán mediante un procedimiento común.

---

## 4. Los datos que necesitamos

Para cada activo y cada vela:

- apertura;
- máximo;
- mínimo;
- cierre;
- volumen;
- timestamp UTC;
- identificador del mercado o metodología de agregación.

Antes de entrenar ningún modelo, se elaborará una ficha de datos con:

- primera y última fecha disponible;
- número esperado y observado de velas;
- huecos temporales;
- duplicados;
- cambios de ticker o de metodología;
- valores nulos o ceros anómalos;
- cobertura común BTC–ETH;
- cobertura adicional de SOL;
- fuente, fecha de descarga y versión del fichero.

La fuente se decidirá en la primera reunión. Coinbase permite obtener velas OHLCV, aunque las peticiones están limitadas y el histórico puede contener intervalos sin operaciones. CoinGecko ofrece histórico de precio y volumen con restricciones de granularidad según el intervalo y el plan. Kraken podrá utilizarse como contraste, pero su endpoint OHLC ordinario solo entrega un número limitado de observaciones recientes.

**Decisión que debe quedar cerrada antes de modelizar:** se utilizará una sola fuente principal. No se mezclarán series de distintos exchanges sin documentar y validar previamente el método de empalme.

---

## 5. La decisión de diseño clave: no modelamos precios, modelamos trayectorias de retornos condicionadas

Esta es la idea central del trabajo.

No se enseñarán al modelo niveles de precio aislados. Se construirán ejemplos del tipo:

> “Así estaba el mercado durante los últimos 30 días” → “Esta fue la trayectoria conjunta de BTC y ETH durante los 30 días siguientes”.

El objetivo principal serán los retornos logarítmicos:

```text
r_t = log(P_t / P_{t-1})
```

El precio se reconstruirá después:

```text
P_t = P_0 · exp(r_1 + r_2 + ... + r_t)
```

Esto evita que el modelo tenga que aprender que un BTC a 10.000 USD y otro a 100.000 USD pueden presentar movimientos porcentuales comparables.

### Condición

La condición resumirá el estado de los 30–60 días anteriores:

- rentabilidad acumulada reciente de cada activo;
- volatilidad realizada de BTC y ETH;
- drawdown actual;
- correlación reciente BTC–ETH;
- variación y nivel normalizado del volumen;
- rango intravela como aproximación a la volatilidad;
- indicador continuo de régimen de mercado.

No se utilizará el precio nominal como variable explicativa principal. Tampoco se impondrán etiquetas subjetivas como “mercado alcista” o “crisis” sin una regla reproducible.

### Objetivo

Cada ejemplo tendrá como objetivo una matriz:

```text
120 pasos × variables generadas
```

Versión mínima:

```text
[retorno_BTC, retorno_ETH]
```

Versión ampliada:

```text
[retorno_BTC, retorno_ETH, rango_BTC, rango_ETH, volumen_BTC, volumen_ETH]
```

El volumen se transformará con logaritmos y se normalizará. Solo se incorporará a la salida si mejora la calidad del modelo sin comprometer el calendario.

---

## 6. Cómo se construirá el dataset

Para cada fecha de inicio válida:

1. se toman los 30–60 días anteriores para calcular la condición;
2. se toman los 30 días siguientes como trayectoria objetivo;
3. se desplaza la ventana y se repite el proceso;
4. se conserva siempre la alineación temporal entre activos.

Las ventanas consecutivas se solapan fuertemente. Por tanto, aunque el dataset contenga muchos ejemplos, estos no son independientes.

### División temporal

El reparto será siempre temporal:

- bloque inicial: entrenamiento;
- bloque intermedio: validación;
- bloque final: prueba;
- purga entre bloques de al menos la longitud máxima de las ventanas utilizadas.

Las fechas exactas se fijarán después de la auditoría de datos y **antes de comparar modelos**. No se cambiará el periodo de prueba porque un modelo haya obtenido malos resultados.

Además de la evaluación con ventanas deslizantes, se realizará una evaluación más exigente con ventanas de prueba no solapadas. Esta segunda evaluación tendrá menos observaciones, pero reflejará mejor la cantidad real de información independiente.

---

## 7. Qué vamos a construir

Un baseline, dos modelos generativos, un tercer modelo opcional, un evaluador común y una aplicación:

| Componente | Qué es |
|---|---|
| Baseline | *Stationary* o *block bootstrap* multivariante. Es el rival principal. |
| Modelo 1 | Autoencoder variacional condicional (CVAE) para trayectorias. |
| Modelo 2 | *Normalizing flow* condicional. |
| Modelo 3 | Difusión temporal o GAN — opcional, solo si lo anterior está cerrado. |
| Evaluador | Batería común de métricas estadísticas, temporales y de riesgo. |
| Aplicación | Revalorización de una cartera BTC–ETH → VaR, ES y drawdown. |

Como control adicional puede incluirse una simulación gaussiana basada en media y covarianza. Su función será mostrar cuánto se pierde al ignorar colas gruesas, volatilidad cambiante y dependencia no lineal.

Una advertencia honesta desde el principio: **es perfectamente posible que el bootstrap gane**. No sería un fracaso. La pregunta del trabajo no es “¿podemos hacer una red neuronal llamativa?”, sino:

> ¿Aporta un modelo generativo no lineal mejoras medibles frente a métodos estadísticos clásicos al reproducir trayectorias, colas, volatilidad y dependencia entre criptoactivos?

---

## 8. El baseline

### Baseline — Block bootstrap multivariante

Se remuestrearán bloques conjuntos de BTC y ETH, no cada activo por separado. Así se preservan:

- correlaciones contemporáneas;
- rachas de volatilidad;
- cierta dependencia temporal;
- episodios conjuntos de subida o caída.

Como mejora, los bloques podrán seleccionarse entre periodos con una condición similar a la actual: volatilidad, tendencia y drawdown próximos.

## 9. Los modelos generativos

### Modelo 1 — CVAE

El encoder recibirá la trayectoria futura real y la condición inicial durante el entrenamiento. El decoder generará una trayectoria completa a partir de:

- una muestra del espacio latente;
- el estado actual del mercado.

Se estudiará:

- dimensión del espacio latente;
- peso del término KL;
- capacidad para evitar trayectorias excesivamente suaves;
- relación entre variables latentes y volatilidad, tendencia o correlación.

### Modelo 2 — Normalizing flow condicional

El flow aprenderá una transformación invertible entre una distribución sencilla y las trayectorias de retornos.

Su ventaja es que permite calcular una densidad exacta **bajo el modelo entrenado**. Esto no debe presentarse como la verdadera probabilidad del mercado, pero sí permite:

- comparar la plausibilidad relativa de escenarios;
- identificar trayectorias anómalas;
- situar shocks históricos o prefijados dentro de la distribución estimada.

Si la dimensión de la trayectoria completa dificulta el entrenamiento, se aplicará el flow sobre una representación temporal reducida obtenida mediante un encoder común o mediante factores previamente definidos. Esta decisión deberá quedar documentada y validada.

### Modelo 3 — Difusión o GAN

Solo se iniciará si:

- los datos están congelados;
- el block bootstrap funciona;
- el evaluador está cerrado;
- CVAE y flow producen resultados reproducibles.

La difusión es preferible si se prioriza estabilidad y diversidad. Una GAN puede ser más rápida, pero presenta mayor riesgo de inestabilidad y *mode collapse*.

---

## 10. El evaluador

No basta con que las trayectorias “parezcan cripto”. Todos los modelos pasarán por la misma batería.

### A. Distribución marginal

Por activo:

- media y desviación típica;
- asimetría y curtosis;
- cuantiles 1%, 5%, 95% y 99%;
- pruebas o distancias entre distribuciones;
- frecuencia y magnitud de movimientos extremos.

### B. Dependencia temporal

- autocorrelación de retornos;
- autocorrelación de retornos absolutos y al cuadrado;
- persistencia de la volatilidad;
- duración de periodos de alta volatilidad;
- agrupamiento de movimientos extremos.

### C. Dependencia entre activos

- correlación BTC–ETH;
- correlación móvil;
- correlación en periodos tranquilos y de estrés;
- probabilidad de caídas conjuntas;
- dependencia en la cola inferior.

### D. Trayectorias

- distribución del retorno acumulado a 30 días;
- máximo drawdown;
- duración del drawdown;
- máximo y mínimo intramensual;
- volatilidad realizada;
- tiempo hasta la pérdida máxima.

### E. Utilidad para riesgo

- cobertura del VaR;
- excepciones del VaR;
- estabilidad del Expected Shortfall;
- pérdidas reales frente a percentiles generados;
- comparación de resultados entre modelos.

### F. Diversidad y memorización

- distancia de cada trayectoria sintética a su vecino real más próximo;
- porcentaje de duplicados o cuasiduplicados;
- cobertura de distintos regímenes;
- prueba discriminativa real frente a sintético;
- evaluación *train on synthetic, test on real* como análisis complementario.

Las métricas se acordarán antes de observar los resultados finales.

---

## 11. La aplicación: stress testing de una cartera

Se definirá una cartera sencilla y transparente:

```text
Valor inicial: 100.000 USD
60% BTC
40% ETH
Sin rebalanceo durante los 30 días
Sin apalancamiento
Sin costes en la versión mínima
```

Cada trayectoria generada producirá:

- valor diario o cada seis horas de la cartera;
- pérdida final a 30 días;
- pérdida máxima durante el periodo;
- máximo drawdown;
- VaR al 95% y 99%;
- Expected Shortfall al 95% y 99%.

### Escenarios de estrés

Se compararán tres tipos:

1. **Históricos:** ventanas reales de crisis reservadas para evaluación.
2. **Prefijados:** caídas conjuntas definidas por el equipo.
3. **Generativos:** trayectorias severas obtenidas de la cola del modelo.

Ejemplos de preguntas:

- ¿Qué pérdida sufriría la cartera en el peor 1% de los escenarios?
- ¿Cómo cambia el riesgo cuando la correlación BTC–ETH aumenta?
- ¿Los modelos generan caídas rápidas y recuperaciones, o solo trayectorias suaves?
- ¿En qué percentil del modelo cae un episodio histórico reservado?
- ¿Difieren mucho el VaR y el ES generativos de los obtenidos por bootstrap?

Los percentiles se interpretarán siempre como probabilidades estimadas **dentro del modelo**, no como frecuencias verdaderas garantizadas.

---

## 12. Reparto

El reparto está diseñado para que nadie espere a nadie. Los tres compartirán desde el primer día:

- definición del dataset;
- interfaz de modelos;
- normalización;
- periodos de entrenamiento, validación y prueba;
- semillas y registro de experimentos.

### Persona A — Datos, baseline y evaluador

Construye el panel temporal alineado, audita los datos, genera las ventanas, implementa el block bootstrap y mantiene el evaluador común.

Es la ruta crítica. Los modelos generativos no pueden evaluarse de forma fiable hasta que exista un pipeline común.

**Compromiso concreto:** entregar durante la primera semana:

- dataset mínimo BTC–ETH;
- split temporal;
- bootstrap funcional;
- primeras métricas marginales, temporales y de cartera.

### Persona B — CVAE y motor de cartera

Entrena el CVAE, analiza el espacio latente y construye el motor que transforma trayectorias de retornos en precios, valores de cartera, pérdidas y drawdowns.

En la segunda mitad:

- calcula VaR y ES;
- implementa las pruebas de cobertura;
- compara riesgo generado y riesgo observado.

### Persona C — Normalizing flow y escenarios de estrés

Entrena el flow condicional y construye el módulo que evalúa la plausibilidad relativa de escenarios históricos y prefijados.

También se encarga de:

- dependencia de cola;
- correlaciones en estrés;
- análisis de trayectorias extremas;
- tercer modelo, únicamente si el flow está cerrado a tiempo.

*(Asignad las letras a nombres en la primera reunión.)*

---

## 13. Calendario

| Semana | Objetivo | Hito |
|---|---|---|
| 28 jul–2 ago | Fuente decidida, datos auditados, universo y splits congelados | **Reunión de arranque y ficha de datos** |
| 3–9 ago | Baseline y primeras versiones de CVAE y flow | **Checkpoint 1: todos pasan por el evaluador mínimo** |
| 10–16 ago | Ajuste y evaluación estadística | **Checkpoint 2 + go/no-go de SOL y tercer modelo** |
| 17–23 ago | Stress testing de la cartera | Modelos congelados |
| 24–28 ago | Experimentos finales y análisis de resultados | **Experimentos congelados** |
| 29 ago–4 sep | Memoria, revisión y presentación | Entrega |

Si el 16 de agosto CVAE, flow o evaluador no son estables, se cancela el tercer modelo. Si el 23 de agosto SOL no está completamente integrado, se elimina de los resultados principales y se presenta, como máximo, en trabajo futuro.

---

## 14. Reglas del juego

1. **La descarga, limpieza y normalización viven en el módulo de datos**, no dentro de cada modelo.
2. **Interfaz común:** todo modelo expone métodos equivalentes a:

   ```text
   fit(X, cond)
   sample(n, cond) -> [n, 120, variables]
   ```

3. **El split es temporal, nunca aleatorio.**
4. **Se purgan las fronteras entre train, validación y test** para evitar que ventanas solapadas filtren información.
5. **BTC y ETH se generan conjuntamente**, nunca con modelos independientes.
6. **No se mezclan exchanges o divisas** sin un estudio previo de compatibilidad.
7. **La cartera, los pesos y las métricas se fijan antes de comparar resultados.**
8. **Las métricas se acuerdan antes de ver los resultados finales.**
9. **Se registran semillas, hiperparámetros, versión de datos y resultados** en metadatos JSON/CSV con checksums; MLflow queda como extensión opcional.
10. **Los escenarios sintéticos no se cuentan como observaciones históricas independientes.**
11. **Congelación de experimentos el 28 de agosto.** La última semana se reserva para redacción y presentación.
12. **El entorno se congela** con Python 3.11.14, dependencias exactas y un `uv.lock` verificado antes de ejecutar tests o notebooks.

---

## 15. Criterios de éxito

El proyecto se considerará satisfactorio si:

- existe un pipeline reproducible de datos a escenarios;
- el entorno puede reconstruirse desde el lock sin resolver versiones nuevas;
- los modelos generan trayectorias válidas y no simples copias;
- se preservan razonablemente colas, volatilidad y dependencia;
- la cartera puede revalorizarse bajo cualquier generador;
- VaR, ES y drawdown se evalúan fuera de muestra;
- se comparan todos los modelos con los mismos criterios;
- las conclusiones reconocen claramente las limitaciones.

No será necesario que el modelo generativo gane en todas las métricas. Un resultado como:

> “El bootstrap ofrece la mejor cobertura del VaR, mientras que el flow representa mejor los cambios de correlación y el CVAE genera mayor diversidad, aunque suaviza las colas”

sería una conclusión válida y defendible.

---

## 16. Lo que este trabajo no hace

- **No predice** el precio futuro de BTC, ETH o SOL.
- **No crea historia económica nueva.**
- **No garantiza** la probabilidad verdadera de una crisis.
- **No convierte** ventanas solapadas en observaciones independientes.
- **No asegura** que los modelos generativos superen al bootstrap.
- **No modela inicialmente** derivados, opciones, liquidaciones, *funding rates* ni apalancamiento.
- **No incorpora** riesgo de custodia, contraparte, hackeo, regulación o pérdida total de un exchange.
- **No genera inicialmente** microestructura de mercado ni libro de órdenes.
- **No debe utilizarse** como sistema real de inversión sin validación adicional.

---

## 17. Versión mínima y extensiones

### Versión mínima obligatoria

- BTC y ETH.
- Retornos a seis horas.
- Trayectorias de 30 días.
- Block bootstrap.
- CVAE.
- Normalizing flow.
- Evaluación común.
- Cartera 60/40.
- VaR, ES y drawdown.

### Extensiones, por orden de prioridad

1. volumen y rango intravela;
2. condicionamiento más rico;
3. SOL como activo con historial más corto;
4. costes y rebalanceo;
5. difusión temporal;
6. variables de derivados: volumen, *funding* y liquidaciones;
7. variables macroeconómicas o de mercado tradicional.

La versión mínima debe quedar completa antes de iniciar cualquiera de estas extensiones.

---

## 18. Referencias de arranque

### Datos

- Coinbase Exchange API — velas históricas:  
  <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles>
- CoinGecko API — histórico dentro de un intervalo:  
  <https://docs.coingecko.com/reference/coins-id-market-chart-range>
- Kraken API — datos OHLC:  
  <https://docs.kraken.com/api-reference/market-data/get-ohlc-data>

### Modelos generativos para series financieras

- *Synthetic Data in Cryptocurrencies using Generative Models* (2026):  
  <https://arxiv.org/abs/2604.16182>
- *Generation of Synthetic Financial Time Series by Diffusion Models* (2024):  
  <https://arxiv.org/abs/2410.18897>
- *Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting* (NeurIPS 2023):  
  <https://github.com/amazon-science/unconditional-time-series-diffusion>
