# Simulación condicional y stress testing de carteras cripto

Pipeline reproducible para generar trayectorias conjuntas de retornos de Bitcoin
y Ethereum a 30 días y evaluar el riesgo de una cartera BTC–ETH.

Este README documenta el dataset común, el entorno, los modelos, las
aplicaciones y las comparaciones reproducibles del proyecto. La planificación
original y el reparto del trabajo están en
[`cripto-generativa-contexto-y-reparto.md`](cripto-generativa-contexto-y-reparto.md).

## Entorno reproducible

El entorno de referencia está congelado con:

- Python 3.11.14, declarado en [`.python-version`](.python-version);
- uv 0.11.24;
- dependencias directas exactas en [`pyproject.toml`](pyproject.toml);
- resolución transitiva, marcadores de plataforma y hashes en
  [`uv.lock`](uv.lock).

El entorno local actual se ha verificado en macOS arm64. El lock también conserva
las resoluciones compatibles disponibles para otras plataformas, pero
TensorFlow y PyTorch pueden utilizar backends de hardware diferentes y no se
promete identidad bit a bit entre CPU, MPS y CUDA.

Desde la raíz del repositorio, el entorno de ejecución de tests, notebooks,
CVAE, flow y GAN se instala con:

```bash
./scripts/setup_environment.sh
source .venv/bin/activate
```

El script ejecuta `uv sync --frozen --all-extras`: falla si `pyproject.toml` y
el lock no coinciden, en lugar de resolver versiones nuevas silenciosamente.
La comprobación puede repetirse en cualquier momento:

```bash
uv lock --check
uv run --frozen python scripts/check_environment.py
uv run --frozen pytest -q
```

No se debe ejecutar `pip install --upgrade` dentro de este entorno. Un cambio de
dependencias es una modificación deliberada del experimento: se actualiza
`pyproject.toml`, se regenera `uv.lock`, se pasan todas las pruebas y se vuelven
a ejecutar únicamente los notebooks cuyos resultados dependan del cambio.

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
importable. Tras preparar el entorno con `scripts/setup_environment.sh`, se
pueden evaluar las distribuciones marginales de sus trayectorias
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

- entorno congelado preparado con `scripts/setup_environment.sh`;
- conexión a Internet únicamente para el primer comando;
- no se necesita clave API de Binance.

Desde la raíz:

```bash
uv run --frozen python scripts/download_binance.py --end 2026-07-31T06:00:00Z
uv run --frozen python scripts/build_btc_eth_panel.py
uv run --frozen python scripts/build_log_returns.py
uv run --frozen python scripts/build_temporal_windows.py
uv run --frozen python scripts/build_condition_features.py
uv run --frozen python scripts/build_temporal_split.py
uv run --frozen python scripts/fit_apply_normalization.py
```

El `--end` es exclusivo y fija el snapshot utilizado por el equipo. No se debe
omitir al regenerar el dataset compartido, porque el valor por defecto añadiría
nuevas velas cerradas y cambiaría muestras, hashes y resultados.

Ejecutar después las pruebas:

```bash
uv run --frozen pytest -q
```

Y verificar las huellas versionadas en cada carpeta:

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
- *Block bootstrap* multivariante: baseline completo en
  [`notebooks/01_block_bootstrap_multivariante.ipynb`](notebooks/01_block_bootstrap_multivariante.ipynb).
- Evaluador común: métricas marginales, temporales, de dependencia BTC–ETH, de
  trayectoria, riesgo, diversidad y memorización disponibles en
  `crypto_generative.evaluation`.
- CVAE condicional: entrenamiento y evaluación ejecutados; el notebook genera
  los artefactos locales de inferencia y escenarios.
- *Normalizing flow* condicional: entrenamiento, densidad exacta, evaluación común
  y experimento *downstream* terminados.
- GAN condicional opcional: entrenamiento, evaluación común y experimento
  *downstream* terminados.
- Aplicación común de cartera y stress testing: implementada para histórico,
  shocks prefijados, bootstrap y artefactos generativos compatibles.
- Generación masiva al último estado disponible: 100.000 escenarios por modelo,
  persistencia por lotes e informe de cartera completados.
- Comparación final consolidada: bootstrap y tres generadores evaluados por
  dimensión, sin agregar objetivos heterogéneos en un score global arbitrario.
- Comparación downstream común: los cuatro métodos evaluados con la misma MLP y
  mezclas `0`, `+25`, `+50` y `+100 %`, manteniendo validación y test reales.

## Aplicación común de cartera y stress testing

La aplicación utiliza una única implementación para todas las fuentes de
escenarios. Revaloriza una cartera inicial de 100.000 USD, con 60 % BTC y 40 %
ETH, sin rebalanceo, apalancamiento ni costes. Para cada conjunto calcula:

- valor de cartera cada seis horas;
- pérdida final y pérdida máxima intraperiodo, en USD y como fracción;
- máximo drawdown;
- VaR y Expected Shortfall al 95 % y 99 %.

El notebook común ejecutado está en
[`notebooks/02_aplicacion_cartera_y_stress.ipynb`](notebooks/02_aplicacion_cartera_y_stress.ipynb).
También puede ejecutarse como aplicación por línea de comandos:

```bash
uv run --frozen python scripts/run_portfolio_stress_test.py
```

La ejecución base compara la distribución histórica de test, los diez peores
drawdowns históricos, tres shocks BTC–ETH prefijados y 5.000 trayectorias del
block bootstrap. Si existen los artefactos `test_scenarios.npz` de CVAE, flow o
GAN bajo `outputs/`, se incorporan automáticamente con las mismas fórmulas. Se
pueden añadir otros artefactos compatibles de forma explícita:

```bash
uv run --frozen python scripts/run_portfolio_stress_test.py \
  --generative mi_modelo=ruta/a/test_scenarios.npz
```

Los resultados se guardan en `outputs/portfolio_stress_test/` como JSON, dos
CSV comparables y figuras. El subconjunto de peores episodios históricos y los
shocks prefijados son ejercicios de severidad seleccionados deliberadamente;
sus percentiles no deben interpretarse como cobertura probabilística.

## Generación masiva al último estado disponible

La aplicación masiva calcula un vector de condición nuevo con las últimas 240
velas completas del panel. Esto es distinto de tomar la última condición de
test, que queda 30 días por detrás del final de los datos porque necesita un
objetivo futuro observado. Después genera por lotes 100.000 trayectorias de cada
modelo y acumula las métricas de la cartera sin mantener todo el cálculo en RAM:

```bash
uv run --frozen python scripts/generate_latest_market_scenarios.py
```

El número de escenarios, tamaño de lote, modelos y dispositivo son configurables:

```bash
uv run --frozen python scripts/generate_latest_market_scenarios.py \
  --n-scenarios 100000 \
  --batch-size 1000 \
  --models cvae normalizing_flow conditional_gan \
  --device auto
```

La vista ejecutada está en
[`notebooks/03_generacion_masiva_ultimo_estado.ipynb`](notebooks/03_generacion_masiva_ultimo_estado.ipynb).
Los artefactos se guardan en `outputs/latest_market_scenarios/`: un `.npy`
mapeable en memoria por modelo, la condición cruda y normalizada, checksums,
metadatos temporales y el informe común de cartera en JSON/CSV. La fecha del
corte se registra de forma explícita: «último» significa el último bloque del
panel proporcionado, no una consulta automática de mercado en vivo.

Generar más trayectorias reduce el error de Monte Carlo dentro de cada modelo,
pero no crea observaciones históricas independientes ni corrige sesgos de
calibración. Para refrescar el estado de mercado primero debe actualizarse el
panel mediante el pipeline de datos y después repetirse esta ejecución.

## Comparación final consolidada

El consolidador reconstruye el informe estructurado del block bootstrap con la
configuración congelada y reúne las métricas de los cuatro métodos:

```bash
uv run --frozen python scripts/build_final_comparison.py
```

La comparación ejecutada está en
[`notebooks/04_comparacion_final_consolidada.ipynb`](notebooks/04_comparacion_final_consolidada.ipynb)
y sus tablas reproducibles en `outputs/final_comparison/`. Se distinguen cuatro
vistas:

- fidelidad estadística, temporal, conjunta, de trayectoria y diversidad;
- utilidad downstream con proporciones sintéticas predefinidas y selección por
  validación;
- VaR/ES y cobertura sobre el test fuera de muestra;
- riesgo masivo bajo la última condición disponible del panel.

No se calcula un ganador global, porque combinar esas dimensiones requeriría
ponderaciones arbitrarias. El block bootstrap ofrece la mejor cobertura VaR y
dependencia de cola; el CVAE destaca en marginales, correlación en estrés,
volatilidad y regímenes; el flow en persistencia, retorno final y drawdown. La
GAN amplía la cobertura geométrica, pero sigue siendo perfectamente distinguible
de los datos reales y presenta errores elevados, por lo que se conserva como
sensibilidad opcional y no como modelo principal.

## Comparación downstream común

El experimento downstream responde directamente al segundo objetivo del taller:
medir qué cambia al entrenar un modelo financiero con distintas cantidades de
datos sintéticos. La tarea, arquitectura e inicialización permanecen fijas:

- entrada: las 14 variables del estado de mercado previo;
- objetivo: máximo drawdown a 30 días de la cartera BTC–ETH 60/40;
- modelo: MLP `64 → 32 → 1`, `dropout=0.10`;
- mezclas: `0`, `+25`, `+50` y `+100 %` de sintéticos respecto al train real;
- validación y test: siempre 100 % reales.

La ejecución completa —block bootstrap, CVAE, Flow y GAN— se regenera con:

```bash
uv run --frozen python scripts/run_common_downstream_comparison.py --device cpu
```

El análisis específico ejecutado y las curvas de convergencia están en
[`notebooks/05_comparacion_downstream_comun.ipynb`](notebooks/05_comparacion_downstream_comun.ipynb).
Sus resultados también forman una dimensión independiente del relato de
[`notebooks/04_comparacion_final_consolidada.ipynb`](notebooks/04_comparacion_final_consolidada.ipynb).
Los CSV, metadatos y PNG ligeros se guardan bajo
`outputs/downstream_common/`. La configuración de `.gitignore` permite
versionarlos; los checkpoints generativos originales continúan fuera de Git.

En test, `+100 %` reduce el MAE frente a `real_only` un 5,16 % con CVAE, un
4,32 % con Flow y un 11,36 % con GAN. El bootstrap empeora el MAE global un
8,68 %, aunque reduce el error del peor decil un 14,31 %. Esta diferencia es
coherente con su naturaleza incondicional: rompe la relación estado–drawdown,
pero aporta episodios severos.

La lectura principal no elige la proporción mirando test. Si se selecciona por
MAE de validación, solo el CVAE escoge augmentación (`+25 %`), con MAE de test
`7,770 %` frente a `7,955 %` de `real_only`; bootstrap, Flow y GAN seleccionan
`real_only`. Además, todos los `R²` siguen siendo negativos. Por tanto, hay
evidencia de mejora relativa para el CVAE, pero no una mejora robusta y general
para cualquier generador o proporción.

## CVAE condicional

El CVAE aprende la distribución de trayectorias futuras condicionada por las 14
variables resumen del estado reciente del mercado:

```text
encoder(target_returns, condition) -> z_mean, z_log_var, z
decoder(z, condition)              -> loc, scale, rho
```

Es una adaptación temporal del VAE convolucional visto en clase. La
configuración implementada usa una salida Student-t bivariante y minimiza:

```text
loss = NLL_Student_t_bivariante(x | loc, scale, rho)
       + beta * KL_regularizada_con_free_bits
```

La Student-t admite colas más gruesas que una Gaussiana y genera BTC y ETH
conjuntamente mediante una correlación aprendida en cada paso. La KL organiza el
espacio latente para muestrear `z ~ N(0, I)` y los *free bits* reducen el riesgo
de colapso posterior temprano. La ejecución documentada utiliza filtros
`(32, 64)`, capa densa 96, latente 8, `beta=0.01`, *free bits* `0.02` y 5 grados
de libertad. El historial de entrenamiento embebido permite comprobar el
comportamiento de la KL en esa ejecución.

La implementación vive en
[`notebooks/CVAE_BTC_ETH.ipynb`](notebooks/CVAE_BTC_ETH.ipynb). Se entrega
completamente ejecutada, con 25 celdas de código y nueve figuras. El notebook
entrena una única configuración y explica su arquitectura, distribución y
pérdida.
La evaluación replica las vistas del baseline de *block bootstrap* para leer con
el mismo formato dependencia temporal, relación BTC–ETH, trayectorias, riesgo y
diversidad. Ambos evalúan ahora sobre las mismas 1.826 trayectorias del test
congelado, por lo que sus métricas permiten una comparación descriptiva fuera de
muestra. El bootstrap genera una distribución agregada y el CVAE 20 escenarios
por condición; además, las ventanas de test se solapan. Por ello la lectura debe
conservar las diferencias entre pronóstico incondicional y condicional y no
tratarse como un backtest con observaciones independientes.

El CVAE utiliza las versiones de TensorFlow y Keras incluidas en el lock común;
no requiere una instalación adicional:

```bash
./scripts/setup_environment.sh
```

El entrenamiento usa solo `train_sample_ids`, selecciona el checkpoint con
`validation_sample_ids` y evalúa el modelo en test deslizante y en 16 anclas
no solapadas. Genera 20 escenarios por condición y entrega los retornos en
unidades originales al `TrajectoryEvaluator` compartido. Así se calculan las
seis familias comunes: marginales, dependencia temporal, dependencia BTC–ETH,
trayectorias, riesgo y diversidad/memorización.

En las salidas ejecutadas del notebook, las desviaciones generadas BTC/ETH son
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

La ejecución genera en `outputs/cvae_best/` los ficheros `encoder.keras`,
`decoder.keras`, `metadata.json` y `test_scenarios.npz`. Este directorio está
ignorado por Git y debe regenerarse ejecutando el notebook cuando no exista en
el entorno local.
