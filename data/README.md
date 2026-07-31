# Datos de mercado

## Fuente principal

El descargador usa el endpoint publico de mercado spot de Binance:

`https://data-api.binance.vision/api/v3/klines`

No requiere API key. Los mercados seleccionados son `BTCUSDT` y `ETHUSDT`, con
velas nativas de 6 horas y timestamps UTC. **USDT no es USD fiat**: se utiliza
como proxy operativo de USD y se conserva el nombre real del activo cotizado en
los CSV y en el manifiesto.

Binance.US dispone de pares literales `BTCUSD` y `ETHUSD`, pero no se usan como
fuente principal porque ofrecen menos historia y pueden contener periodos sin
negociacion. No se deben empalmar ambas fuentes sin un analisis previo.

## Descarga reproducible

Desde la raiz del proyecto:

```bash
python3 scripts/download_binance.py
```

El intervalo por defecto es `[2017-01-01, inicio de la vela actual)`. Por tanto,
solo se guardan velas cerradas. Se puede congelar una fecha de corte concreta:

```bash
python3 scripts/download_binance.py --end 2026-07-31T00:00:00Z
```

Se generan:

- `data/raw/binance/btcusdt_6h.csv`;
- `data/raw/binance/ethusdt_6h.csv`;
- `data/raw/binance/manifest_6h.json`;
- `data/raw/binance/SHA256SUMS`, generado al congelar una descarga concreta.

El manifiesto registra fuente, mercado, divisa base y cotizada, intervalo,
fechas solicitadas y observadas, fecha de descarga, huecos, duplicados y valores
de precio o volumen anomalos. Los ficheros brutos estan ignorados por Git porque
son regenerables y su version queda definida por el manifiesto.

## Panel limpio BTC-ETH

Para validar y alinear ambos activos en un calendario regular de seis horas:

```bash
python3 scripts/build_btc_eth_panel.py
```

El resultado vive en `data/processed/binance/`. Los huecos se conservan como
filas explicitas, con campos numericos vacios y `is_complete=0`. Las velas de
duracion anomala conservan sus valores para auditoria, pero quedan marcadas con
`*_duration_valid=0` e `is_complete=0`. `expected_close_time_utc` representa el
final teorico de la rejilla y `*_source_close_time_utc`, el cierre realmente
informado por Binance. No se imputan precios ni volumen. Las futuras ventanas de
entrenamiento que atraviesen filas incompletas deben excluirse.

## Retornos logaritmicos

Para calcular los retornos conjuntos:

```bash
python3 scripts/build_log_returns.py
```

El resultado se guarda en `data/features/binance/`. Se aplica
`r_t = log(close_t / close_(t-1))`. La primera fila y cualquier retorno cuya
vela actual o anterior tenga `is_complete=0` quedan vacios y marcados con
`returns_valid=0`; nunca se salta un hueco para calcular un retorno de falsa
frecuencia.

## Ventanas temporales

Para construir condiciones de 60 dias y objetivos de 30 dias:

```bash
python3 scripts/build_temporal_windows.py
```

El fichero `.npz` contiene matrices `condition_returns [n, 240, 2]` y
`target_returns [n, 120, 2]`, en orden de activos `[BTC, ETH]`. El CSV de indice
registra las fronteras temporales de cada muestra. Se usa un desplazamiento de
una vela, por lo que las muestras se solapan fuertemente y no son observaciones
independientes. Toda ventana que toque un retorno no valido se descarta.

## Vector resumido de condicion

Para resumir los 60 dias anteriores a cada objetivo:

```bash
python3 scripts/build_condition_features.py
```

Se generan 14 variables: por activo, retorno acumulado, volatilidad anualizada,
drawdown actual, nivel normalizado y cambio del volumen, y rango intravela;
ademas se calculan la correlacion BTC-ETH y un indicador continuo de regimen.
La tabla conserva las fronteras temporales y `sample_id`. No se aplica una
normalizacion global para evitar filtracion de informacion antes del split.

## Split temporal con purga

Para congelar entrenamiento, validacion y prueba:

```bash
python3 scripts/build_temporal_split.py
```

La asignacion usa `target_start_utc`. Se purgan 90 dias de inicios de objetivo
entre entrenamiento-validacion y validacion-prueba, equivalentes a la ventana
completa de 60 dias de condicion y 30 de objetivo. El constructor verifica que
los intervalos brutos conservados no compartan timestamps. Las muestras
purgadas permanecen identificadas en el indice, pero no se usan para ajustar ni
evaluar modelos.

## Normalizacion ajustada en entrenamiento

Para ajustar y aplicar la normalizacion sin filtracion temporal:

```bash
python3 scripts/fit_apply_normalization.py
```

Las 14 condiciones se estandarizan con las muestras de entrenamiento. Los
retornos se estandarizan por activo usando timestamps unicos cubiertos por las
ventanas de entrenamiento, evitando ponderarlos repetidamente por el
solapamiento. Validacion, prueba y purgas reciben exactamente la misma
transformacion, sin reajuste ni recorte.
