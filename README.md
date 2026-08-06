# Simulación condicional y stress testing de carteras cripto

Pipeline reproducible para generar trayectorias conjuntas de retornos de Bitcoin
y Ethereum a 30 días y evaluar el riesgo de una cartera BTC–ETH.

Este README explica principalmente cómo deben consumir los datos las personas
encargadas del CVAE y del *normalizing flow*. La planificación completa y el
reparto del trabajo están en
[`cripto-generativa-contexto-y-reparto.md`](cripto-generativa-contexto-y-reparto.md).

## Resumen del dataset

| Propiedad | Valor |
|---|---|
| Fuente | Binance Spot, API pública |
| Mercados reales | `BTCUSDT`, `ETHUSDT` |
| Divisa cotizada | USDT — no es USD fiat |
| Frecuencia | Velas de 6 horas, UTC, mercado 24/7 |
| Cobertura bruta | 17/08/2017 00:00 – 31/07/2026 00:00 UTC |
| Condición histórica | 60 días = 240 retornos |
| Objetivo generado | 30 días = 120 retornos |
| Orden de activos | índice `0 = BTC`, índice `1 = ETH` |
| Muestras válidas | 9.096 |
| Variables de condición | 14 |
| Split | Temporal, sin aleatorización y con purgas de 90 días |

Los nombres habituales `BTC-USD` y `ETH-USD` se usan para describir el universo
económico. Los mercados descargados son realmente `BTCUSDT` y `ETHUSDT`; esta
diferencia se conserva explícitamente en los datos y metadatos.

## Qué se necesita para entrenar modelos

Para entrenar el CVAE o el flow solo son imprescindibles estos tres artefactos,
incluidos directamente en el repositorio:

```text
data/normalized/binance/
└── btc_eth_6h_c240_t120_train_normalized.npz

data/splits/binance/
├── btc_eth_6h_c240_t120_purged_split.npz
└── btc_eth_6h_c240_t120_purged_split_index.csv
```

El primer fichero contiene las entradas y objetivos normalizados. El segundo
contiene los `sample_id` asignados a entrenamiento, validación, prueba y purgas.
El CSV permite relacionar cada muestra con fechas reales.

También se incluye `data/processed/binance/btc_eth_6h_panel.csv`, necesario para
recuperar los precios iniciales al reconstruir trayectorias de precios o valorar
la cartera. Después de clonar el repositorio no es necesario regenerar datos
para comenzar a entrenar.

### Carga recomendada

Ejecutar desde la raíz del proyecto:

```python
from pathlib import Path

import numpy as np


ROOT = Path.cwd()

normalized_path = (
    ROOT
    / "data"
    / "normalized"
    / "binance"
    / "btc_eth_6h_c240_t120_train_normalized.npz"
)
split_path = (
    ROOT
    / "data"
    / "splits"
    / "binance"
    / "btc_eth_6h_c240_t120_purged_split.npz"
)

with np.load(normalized_path, allow_pickle=False) as data:
    condition = data["condition_features"]
    condition_history = data["condition_returns"]
    target = data["target_returns"]
    feature_names = data["feature_names"]
    assets = data["assets"]
    sample_ids = data["sample_ids"]
    condition_mean = data["condition_feature_mean"]
    condition_scale = data["condition_feature_scale"]
    return_mean = data["return_mean"]
    return_scale = data["return_scale"]

with np.load(split_path, allow_pickle=False) as split:
    train_ids = split["train_sample_ids"]
    validation_ids = split["validation_sample_ids"]
    test_ids = split["test_sample_ids"]

condition_train = condition[train_ids]
condition_validation = condition[validation_ids]
condition_test = condition[test_ids]

target_train = target[train_ids]
target_validation = target[validation_ids]
target_test = target[test_ids]

print(condition_train.shape)       # (4710, 14)
print(condition_validation.shape)  # (1832, 14)
print(condition_test.shape)        # (1826, 14)
print(target_train.shape)          # (4710, 120, 2)
print(assets.tolist())             # ['BTC', 'ETH']
```

Correspondencia con la interfaz común acordada:

```python
model.fit(X=target_train, cond=condition_train)

generated_normalized = model.sample(
    n=1_000,
    cond=condition_test[some_sample],
)
```

Los métodos concretos pueden usar otros nombres, pero deben respetar estas
formas:

```text
cond:   [n_muestras, 14]
X:      [n_muestras, 120, 2]
sample: [n_escenarios, 120, 2]
```

`condition_history` tiene forma `[9096, 240, 2]`. No es la condición principal
del modelo mínimo, pero queda disponible para arquitecturas con encoder temporal
o para análisis adicionales.

## Significado de los arrays normalizados

El fichero
`data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz` contiene:

| Array | Forma | Significado |
|---|---:|---|
| `condition_features` | `(9096, 14)` | Estado resumido de los 60 días anteriores |
| `condition_returns` | `(9096, 240, 2)` | Historia conjunta de retornos de la condición |
| `target_returns` | `(9096, 120, 2)` | Trayectoria futura conjunta que aprende el modelo |
| `feature_names` | `(14,)` | Nombres y orden de las variables de condición |
| `assets` | `(2,)` | Orden fijo `['BTC', 'ETH']` |
| `sample_ids` | `(9096,)` | Identificador estable entre todos los artefactos |
| `condition_feature_mean` | `(14,)` | Media de las condiciones de entrenamiento |
| `condition_feature_scale` | `(14,)` | Desviación de las condiciones de entrenamiento |
| `return_mean` | `(2,)` | Media BTC/ETH ajustada con retornos de entrenamiento |
| `return_scale` | `(2,)` | Desviación BTC/ETH ajustada con entrenamiento |

Todos los arrays numéricos utilizan `float64`. Se pueden convertir a `float32`
al crear tensores para PyTorch o TensorFlow:

```python
import torch

x_train = torch.from_numpy(target_train).float()
c_train = torch.from_numpy(condition_train).float()
```

No se deben intercambiar los ejes de activo ni entrenar BTC y ETH por separado.
La dependencia conjunta forma parte del problema.

## Las 14 variables de condición

El orden exacto siempre debe leerse desde `feature_names`; actualmente es:

| Índice | Variable | Interpretación |
|---:|---|---|
| 0 | `btc_cumulative_log_return_60d` | Suma de retornos logarítmicos BTC en 60 días |
| 1 | `btc_realized_volatility_ann_60d` | Volatilidad BTC anualizada |
| 2 | `btc_current_drawdown_60d` | Caída del valor final respecto al máximo de la ventana |
| 3 | `btc_log_volume_z_7d` | Nivel reciente del log-volumen respecto a los 60 días |
| 4 | `btc_log_volume_change_7d` | Log-volumen de los últimos 7 días menos los 7 anteriores |
| 5 | `btc_mean_log_range_7d` | Media reciente de `log(high / low)` |
| 6–11 | equivalentes ETH | Las mismas seis variables para Ethereum |
| 12 | `btc_eth_correlation_30d` | Correlación de Pearson de los últimos 30 días |
| 13 | `joint_trend_regime_score_60d` | Tendencia conjunta escalada por volatilidad |

Las variables anteriores aparecen normalizadas en el dataset final. Sus valores
originales están en:

```text
data/features/binance/btc_eth_6h_c240_condition_summary.npz
data/features/binance/btc_eth_6h_c240_condition_summary.csv
```

La normalización es:

```text
z = (x - media_entrenamiento) / desviación_entrenamiento
```

Se ajustó exclusivamente con `train_sample_ids`. Validación, prueba y purgas se
transformaron con esos mismos parámetros, sin reajuste ni *clipping*. Por ello,
es correcto que validación o prueba no tengan media cero y que algunos valores
queden fuera del rango observado en entrenamiento.

## Retornos, precios y transformación inversa

El objetivo contiene retornos logarítmicos de seis horas:

```text
r_t = log(P_t / P_(t-1))
```

Antes de calcular métricas financieras o reconstruir precios hay que invertir
la normalización:

```python
generated_returns = generated_normalized * return_scale + return_mean
```

La reconstrucción de precios es:

```python
log_paths = np.cumsum(generated_returns, axis=1)
price_paths = initial_prices[None, None, :] * np.exp(log_paths)
```

`initial_prices` debe contener el cierre BTC/ETH de `condition_end_utc` para la
muestra condicionante. Se puede obtener combinando:

```text
data/splits/binance/btc_eth_6h_c240_t120_purged_split_index.csv
data/processed/binance/btc_eth_6h_panel.csv
```

No se deben reconstruir precios a partir de retornos todavía normalizados.

## Evaluador común

El primer bloque del evaluador compartido ya está disponible como módulo
importable. Tras instalar el proyecto con `python -m pip install -e .`, se pueden evaluar las distribuciones marginales de sus trayectorias
con la misma implementación:

```python
import pandas as pd

from crypto_generative.evaluation import (
    CrossAssetDependenceConfig,
    DiversityMemorizationConfig,
    RiskMetricsConfig,
    TemporalDependenceConfig,
    TrajectoryEvaluator,
    TrajectoryMetricsConfig,
)


# target_validation procede del ejemplo de carga anterior.
reference_returns = target_validation * return_scale + return_mean
training_returns = target_train * return_scale + return_mean

# Salida condicional del modelo: [condiciones, draws, 120 pasos, 2 activos].
generated_conditional_returns = generated_normalized * return_scale + return_mean

# Las familias descriptivas consumen un lote 3D.
generated_returns = generated_conditional_returns.reshape(
    -1,
    generated_conditional_returns.shape[-2],
    generated_conditional_returns.shape[-1],
)

evaluator = TrajectoryEvaluator(assets=assets.tolist())
marginal_report = evaluator.evaluate_marginals(
    reference_paths=reference_returns,
    candidate_paths=generated_returns,
)

table = pd.DataFrame(marginal_report.to_records())
print(table)

temporal_report = evaluator.evaluate_temporal_dependence(
    reference_paths=reference_returns,
    candidate_paths=generated_returns,
    config=TemporalDependenceConfig(
        max_lag=20,                    # 5 días en velas de 6h
        volatility_window=20,          # volatilidad móvil de 5 días
        high_volatility_quantile=0.90,
        extreme_quantile=0.99,
        extreme_clustering_window=4,   # extremos durante las 24h previas
    ),
)

temporal_table = pd.DataFrame(temporal_report.to_records())
print(temporal_table)

cross_asset_report = evaluator.evaluate_cross_asset_dependence(
    reference_paths=reference_returns,
    candidate_paths=generated_returns,
    config=CrossAssetDependenceConfig(
        rolling_window=20,       # correlación móvil de 5 días
        stress_quantile=0.90,
        joint_drop_quantile=0.05,
        lower_tail_quantile=0.05,
    ),
)

cross_asset_table = pd.DataFrame(cross_asset_report.to_records())
print(cross_asset_table)

trajectory_report = evaluator.evaluate_trajectories(
    reference_paths=reference_returns,
    candidate_paths=generated_returns,
    config=TrajectoryMetricsConfig(periods_per_year=4 * 365),
)

trajectory_table = pd.DataFrame(trajectory_report.to_records())
print(trajectory_table)

risk_report = evaluator.evaluate_risk(
    reference_paths=reference_returns,
    candidate_paths=generated_conditional_returns,
    config=RiskMetricsConfig(
        confidence_levels=(0.95, 0.99),
        portfolio_weights=(0.60, 0.40),
        portfolio_name="portfolio_60_40",
        es_stability_repetitions=100,
        es_stability_sample_size=1_000,
        random_state=42,
    ),
)

risk_table = pd.DataFrame(risk_report.to_records())
print(risk_table)

diversity_report = evaluator.evaluate_diversity_and_memorization(
    reference_paths=reference_returns,
    candidate_paths=generated_returns,
    training_paths=training_returns,
    config=DiversityMemorizationConfig(
        max_paths_per_set=2_000,
        projection_dimensions=24,
        neighbor_candidates=8,
        near_memorization_quantile=0.01,
        coverage_radius_quantile=0.95,
        discriminator_repetitions=5,
        random_state=42,
    ),
)

diversity_table = pd.DataFrame(diversity_report.to_records())
print(diversity_table.T)
```

El contrato común es deliberadamente estricto:

- retornos logarítmicos **desnormalizados**;
- forma `[trayectorias, tiempo, activos]`, salvo el candidato condicional 4D de
  `evaluate_risk()`;
- horizonte y orden de activos iguales en referencia y candidato;
- número de trayectorias real y sintético puede ser distinto;
- no se admiten `NaN` ni infinitos.

Si el modelo genera un tensor
`[condiciones, draws, tiempo, activos]`, para las familias marginal, temporal,
BTC–ETH y trayectoria hay que combinar únicamente las dos primeras dimensiones,
como hace `generated_returns` en el ejemplo anterior.

Para `evaluate_risk()` debe conservarse el tensor 4D. El evaluador calculará un
VaR y un ES distintos para cada condición y comparará cada observación real con
su propia distribución predictiva. Si recibe un candidato 3D, como el
bootstrap, aplicará una única distribución de riesgo agregada a todas las
observaciones reales.

La familia marginal calcula por activo media, desviación típica, asimetría,
curtosis en exceso, cuantiles 1/5/50/95/99 %, frecuencia y magnitud de extremos,
y distancia Wasserstein-1 absoluta y normalizada.

La familia temporal calcula ACF de retornos, retornos absolutos y cuadrados;
persistencia de volatilidad; frecuencia y duración de episodios de alta
volatilidad; y agrupamiento de extremos. Los umbrales de alta volatilidad y
movimiento extremo se ajustan **solo en la referencia** y se aplican sin
reajuste al candidato. Las ACF y las duraciones respetan siempre los límites de
cada trayectoria.

La familia BTC–ETH compara correlación contemporánea, distribución de
correlaciones móviles, correlación en calma y estrés, frecuencia de caídas
conjuntas y dependencia en la cola inferior. La probabilidad de caída conjunta
usa umbrales del 5 % ajustados solo en la referencia. La dependencia de cola
usa el percentil propio de cada lote para aislar mejor la estructura de
dependencia respecto de los errores marginales.

La familia de trayectorias compara la distribución del retorno acumulado a 30
días, volatilidad realizada anualizada, máximo drawdown, duración máxima del
drawdown, máximo y mínimo dentro del horizonte y tiempo hasta el valor mínimo.
Las duraciones se expresan en pasos de seis horas y los retornos y drawdowns
como fracciones de riqueza inicial.

La familia de riesgo calcula VaR y Expected Shortfall al 95 % y 99 % para BTC,
ETH y una cartera inicial 60/40 sin rebalanceo. Incluye errores frente a la
referencia, número y tasa de excepciones, error de cobertura, estabilidad del ES
mediante remuestreo y percentiles que las pérdidas reales ocupan dentro del
modelo. Las pérdidas se expresan como fracción positiva de la riqueza inicial.

La familia de diversidad y memorización calcula duplicados, vecinos cercanos
entre escenarios, coincidencias exactas y cercanía anómala a train, cobertura
de validación, proporciones de regímenes de volatilidad y exactitud de un
discriminador lineal real–sintético. Una exactitud próxima a 0,5 indica que el
discriminador no separa ambos lotes; valores altos indican diferencias
sistemáticas.

Las distancias son RMSE sobre retornos estandarizados. La búsqueda usa una
proyección aleatoria para proponer vecinos y recalcula la distancia original de
los mejores candidatos. Para limitar coste y memoria se evalúa como máximo el
número de trayectorias indicado por `max_paths_per_set`, mediante una muestra
reproducible. El informe conserva los tamaños realmente utilizados.

La evaluación *train on synthetic, test on real* no forma parte de esta API
genérica: necesita definir una tarea predictiva, etiquetas y un estimador común,
por lo que debe plantearse como experimento complementario separado.

Durante el desarrollo se debe pasar **validación** como referencia. Prueba se
utilizará únicamente cuando el modelo y su configuración estén congelados. El
solapamiento de ventanas no desaparece por usar el evaluador: no deben
interpretarse sus filas como historias independientes al construir intervalos
de confianza.

## Split temporal y purgas

La clave de asignación es `target_start_utc`:

| Bloque | Regla | Muestras |
|---|---|---:|
| Entrenamiento | antes de `2023-07-01` | 4.710 |
| Purga train–validation | `2023-07-01` a `2023-09-30` | 368 |
| Validación | `2023-10-01` a `2024-12-31` | 1.832 |
| Purga validation–test | `2025-01-01` a `2025-03-31` | 360 |
| Prueba | desde `2025-04-01` | 1.826 |

Las purgas cubren 90 días de anclajes: 60 días de condición y 30 días de
objetivo. Los intervalos brutos conservados no comparten timestamps.

Reglas obligatorias:

1. No mover las fechas porque un modelo obtenga malos resultados.
2. No usar las 728 muestras purgadas para entrenamiento, selección o evaluación.
3. No ajustar normalizadores con validación o prueba.
4. Se puede barajar `train_ids` dentro del *DataLoader* después de aplicar el
   split; no se puede crear un split aleatorio nuevo.
5. Validación sirve para arquitectura e hiperparámetros. Prueba se reserva para
   la evaluación final.

Las ventanas consecutivas se desplazan solo seis horas y se solapan mucho. Las
9.096 muestras no representan 9.096 historias económicas independientes. Para
intervalos de confianza y algunas métricas finales se necesitará además una
evaluación con ventanas no solapadas.

## Capas de datos y procedencia

```text
Binance API
  └── data/raw/binance/                    velas OHLCV originales
       └── data/processed/binance/         panel BTC-ETH regular y auditado
            └── data/features/binance/     retornos y condición resumida
                 └── data/windows/binance/ ventanas de 60d → 30d
                      ├── data/splits/binance/     IDs temporales purgados
                      └── data/normalized/binance/ arrays listos para modelos
```

### Velas brutas

```text
data/raw/binance/btcusdt_6h.csv
data/raw/binance/ethusdt_6h.csv
data/raw/binance/manifest_6h.json
```

Cada fila contiene apertura, máximo, mínimo, cierre, volumen, volumen cotizado,
número de operaciones, compras del *taker*, timestamps UTC y procedencia del
mercado.

### Panel limpio

```text
data/processed/binance/btc_eth_6h_panel.csv
```

Tiene 13.081 timestamps regulares. Se detectaron seis velas completamente
ausentes y doce velas truncadas, comunes a BTC y ETH. No se imputaron valores.
Las 18 filas afectadas tienen `is_complete=0` y no participan en ninguna ventana
válida.

### Retornos

```text
data/features/binance/btc_eth_6h_log_returns.csv
```

Contiene 13.050 retornos conjuntos válidos. Un retorno solo existe cuando la
vela actual y la anterior son completas; nunca se salta un hueco para construir
un falso retorno de seis horas.

### Ventanas originales sin normalizar

```text
data/windows/binance/btc_eth_6h_c240_t120_s1.npz
data/windows/binance/btc_eth_6h_c240_t120_s1_index.csv
```

Contienen `condition_returns` y `target_returns` en unidades originales. Son
útiles para evaluación, inversión de transformaciones y baselines. Los modelos
neuronales deben consumir por defecto el artefacto normalizado.

## Reproducir todos los datos desde cero — opcional

Requisitos:

- Python 3.9 o posterior;
- NumPy 1.23 o posterior;
- conexión a Internet únicamente para el primer comando;
- no se necesita clave API de Binance.

Desde la raíz:

```bash
python3 scripts/download_binance.py --end 2026-07-31T06:00:00Z
python3 scripts/build_btc_eth_panel.py
python3 scripts/build_log_returns.py
python3 scripts/build_temporal_windows.py
python3 scripts/build_condition_features.py
python3 scripts/build_temporal_split.py
python3 scripts/fit_apply_normalization.py
```

El `--end` es exclusivo y fija el snapshot utilizado por el equipo. No se debe
omitir al regenerar el dataset compartido, porque el valor por defecto añadiría
nuevas velas cerradas y cambiaría muestras, hashes y resultados.

Ejecutar después las pruebas:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Y verificar las huellas publicadas en cada carpeta:

```bash
cd data/normalized/binance
shasum -a 256 -c btc_eth_6h_c240_t120_train_normalized_SHA256SUMS
```

## Datasets incluidos en Git

El repositorio versiona un snapshot mínimo y congelado para que las personas B y
C puedan entrenar inmediatamente después de clonarlo:

```text
data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz
data/normalized/binance/btc_eth_6h_c240_t120_train_normalized_manifest.json
data/splits/binance/btc_eth_6h_c240_t120_purged_split.npz
data/splits/binance/btc_eth_6h_c240_t120_purged_split_index.csv
data/splits/binance/btc_eth_6h_c240_t120_purged_split_manifest.json
data/processed/binance/btc_eth_6h_panel.csv
data/processed/binance/manifest_6h.json
```

Sus ficheros `SHA256SUMS` también se versionan. En conjunto ocupan alrededor de
7,2 MB, por lo que no requieren Git LFS. Los datos brutos y los artefactos
intermedios redundantes permanecen ignorados y se pueden reconstruir con los
scripts anteriores.

Este snapshot no debe reemplazarse silenciosamente. Una actualización de datos
requiere nueva fecha de corte, regenerar todo el pipeline, revisar la auditoría,
actualizar los checksums y comunicar una nueva versión al equipo.

## Errores que deben evitarse

- Tratar USDT como si fuera USD fiat sin documentarlo.
- Cambiar el orden `[BTC, ETH]` de la última dimensión.
- Ajustar otra normalización dentro de validación o prueba.
- Entrenar con `purge_*_sample_ids`.
- Calcular métricas financieras con retornos normalizados.
- Interpretar ventanas solapadas como observaciones independientes.
- Cambiar las fechas de prueba tras observar resultados.
- Generar BTC y ETH mediante modelos independientes.
- Usar el precio nominal como objetivo del modelo.

## Estado actual

- Descarga y auditoría: completadas.
- Panel BTC–ETH: completado.
- Retornos logarítmicos: completados.
- Ventanas 60 → 30 días: completadas.
- Vector de condición: completado.
- Split temporal con purga: congelado.
- Normalización ajustada en entrenamiento: completada.
- *Block bootstrap* multivariante: primera versión didáctica en
  [`notebooks/01_block_bootstrap_multivariante.ipynb`](notebooks/01_block_bootstrap_multivariante.ipynb).
- Evaluador común: métricas marginales, temporales, de dependencia BTC–ETH, de
  trayectoria, riesgo, diversidad y memorización disponibles en
  `crypto_generative.evaluation`.
- CVAE condicional: búsqueda completa, evaluación y artefactos terminados.
- Próximo componente opcional: comparar contra el *normalizing flow* del equipo.

## CVAE condicional

El CVAE aprende la distribución de trayectorias futuras condicionada por las 14
variables resumen del estado reciente del mercado:

```text
encoder(target_returns, condition) -> z_mean, z_log_var, z
decoder(z, condition)              -> loc, scale, rho
```

Es la adaptación temporal del VAE convolucional visto en clase. El notebook
demuestra por qué un decoder MSE no basta para retornos financieros: aproxima la
media y genera trayectorias demasiado suaves. La búsqueda compara MSE, Gaussiana
bivariante y Student-t bivariante. Para las salidas probabilísticas se minimiza:

```text
loss = NLL_Student_t_bivariante(x | loc, scale, rho)
       + beta * KL_regularizada_con_free_bits
```

La Student-t conserva colas gruesas y genera BTC y ETH conjuntamente mediante
una correlación aprendida en cada paso. La KL organiza el espacio latente para
muestrear `z ~ N(0, I)` y los *free bits* evitan el colapso posterior temprano.

La búsqueda ejecutada compara 39 configuraciones: distribución, capacidad,
latentes 4–16, `beta`, *free bits*, grados de libertad, condición resumen,
Conv1D, GRU, híbrida y pérdida acumulada. Cada candidato usa un barajado
determinista; el top 3 se repite con semillas 7, 42 y 123. El ganador se elige
por `media + desviación` del score de validación, no por una ejecución aislada.

El ganador actual es `student_medium_l8_b01`: Student-t bivariante, filtros
`(32, 64)`, capa densa 96, latente 8, `beta=0.01`, *free bits* `0.02`, 5 grados
de libertad y condición resumen. El KL de validación permanece activo (~3,55 de
media entre semillas), por lo que no hay evidencia de colapso posterior.

La implementación vive en
[`notebooks/CVAE_BTC_ETH.ipynb`](notebooks/CVAE_BTC_ETH.ipynb). Se entrega
completamente ejecutada, con 25 celdas de código y nueve figuras. El notebook
entrena únicamente el ganador; la sección teórica resume la búsqueda de 39
configuraciones y explica extensamente su arquitectura, distribución y pérdida.
La evaluación replica las vistas del baseline de *block bootstrap* para leer con
el mismo formato dependencia temporal, relación BTC–ETH, trayectorias, riesgo y
diversidad. El notebook advierte que el baseline publicado usa validación y el
CVAE usa test, por lo que sus cifras no constituyen aún una comparación directa.

Instalación:

```bash
python3 -m pip install -e '.[cvae]'
```

El entrenamiento usa solo `train_sample_ids`, selecciona con
`validation_sample_ids` y evalúa el ganador en test deslizante y en 16 anclas
no solapadas. Genera 20 escenarios por condición y entrega los retornos en
unidades originales al `TrajectoryEvaluator` compartido. Así se calculan las
seis familias comunes: marginales, dependencia temporal, dependencia BTC–ETH,
trayectorias, riesgo y diversidad/memorización.

En la ejecución publicada, las desviaciones generadas BTC/ETH son
`0.01093/0.01654`, frente a `0.01091/0.01707` reales, y el error absoluto de
correlación contemporánea es `0.023`. Las trayectorias son 100 % únicas, no hay
coincidencias exactas con train y la cobertura de la referencia alcanza el
94.8 %. Aun así, el Wasserstein normalizado del retorno final es `0.436/0.240`,
la dependencia de cola inferior tiene un error de `0.165` y un discriminador
distingue real de sintético con 74.5 % de acierto.

El riesgo condicional no está todavía calibrado: en VaR 95 % aparecen alrededor
de 20–22 % de excepciones en vez del 5 % esperado. El VaR/ES 99 % es además
exploratorio porque 20 draws por condición no resuelven suficientemente esa
cola. Las 16 anclas no solapadas sirven como sensibilidad, no como estimación
precisa de eventos extremos.

Los artefactos quedan en `outputs/cvae_best/`: `encoder.keras`, `decoder.keras`,
`metadata.json` y `test_scenarios.npz`. Los CSV de búsqueda y estabilidad se
conservan como evidencia histórica del benchmark, pero el notebook no vuelve a
entrenar esos candidatos.
