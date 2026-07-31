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
- Próximo componente: *block bootstrap* multivariante.
