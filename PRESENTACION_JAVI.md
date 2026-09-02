# Guía de la presentación de Javi

Esta guía sirve para entender el proyecto completo y, especialmente, para
preparar la parte de Javier: el **CVAE condicionado** y la **conclusión final**
de las diapositivas 6 y 7.

## 1. El proyecto en una frase

Queremos generar muchos futuros plausibles de Bitcoin y Ethereum para estudiar
el riesgo de una cartera y comprobar si los datos sintéticos también ayudan a
entrenar un modelo predictivo.

No intentamos adivinar un único precio futuro. Intentamos aprender una
**distribución de posibles trayectorias**: futuros tranquilos, volátiles,
positivos, negativos y extremos.

## 2. ¿Qué problema intentamos resolver?

Para medir riesgo financiero hacen falta ejemplos de situaciones normales y de
crisis. El problema es que:

- los episodios extremos son escasos en los datos históricos;
- Bitcoin y Ethereum no siempre se comportan igual;
- su relación puede cambiar precisamente durante las crisis;
- repetir únicamente el pasado limita la variedad de escenarios disponibles.

Por eso estudiamos si varios modelos generativos pueden crear trayectorias
sintéticas conjuntas de BTC y ETH que sean:

1. parecidas a las observadas en el mercado;
2. útiles para calcular pérdidas y eventos extremos;
3. útiles como datos adicionales para una tarea de predicción.

La pregunta principal es:

> ¿Los modelos generativos neuronales aportan mejoras medibles frente a un
> método estadístico sencillo?

## 3. Los datos, explicados de forma sencilla

Usamos velas de mercado spot de Binance para `BTCUSDT` y `ETHUSDT`, con una
frecuencia de 6 horas. USDT se utiliza como aproximación operativa al dólar,
pero no es USD fiat.

| Elemento | Valor |
|---|---:|
| Periodo | 17/08/2017-31/07/2026 |
| Frecuencia | 6 horas |
| Contexto observado | 60 días = 240 pasos |
| Futuro generado | 30 días = 120 pasos |
| Activos | BTC y ETH conjuntamente |
| Variables de condición | 14 |
| Ventanas válidas | 9.096 |
| Entrenamiento | 4.710 |
| Validación | 1.832 |
| Prueba | 1.826 |
| Ventanas eliminadas mediante purga | 728 |

Cada ejemplo puede imaginarse así:

```text
60 días anteriores                         30 días siguientes
contexto del mercado       ->              futuro BTC + ETH
240 retornos por activo                    120 retornos por activo
14 señales resumidas                       trayectoria que queremos generar
```

### Las 14 variables de condición

Para cada activo calculamos seis señales:

1. retorno acumulado de 60 días;
2. volatilidad anualizada de 60 días;
3. drawdown actual de 60 días;
4. nivel normalizado del volumen reciente;
5. cambio reciente del volumen;
6. rango medio entre máximo y mínimo de la vela.

Eso produce 12 variables. Añadimos otras dos conjuntas:

13. correlación reciente entre BTC y ETH;
14. indicador continuo del régimen de mercado.

Estas variables responden a una idea sencilla: un futuro generado después de
un periodo tranquilo no debería tener necesariamente la misma distribución que
un futuro generado en plena crisis.

### División temporal y prevención de fugas

El orden temporal se conserva: primero entrenamiento, después validación y al
final prueba. Entre los conjuntos se eliminan ventanas mediante una purga para
que sus intervalos brutos no compartan fechas.

La normalización se ajusta exclusivamente con entrenamiento. Validación y test
reciben esa misma transformación, sin recalcularla. Así evitamos que el modelo
obtenga indirectamente información del futuro.

Las ventanas avanzan una vela cada vez y se solapan mucho. Por tanto, las 9.096
ventanas no equivalen a 9.096 episodios económicos independientes. El proyecto
también comprueba 16 condiciones no solapadas como análisis de sensibilidad.

## 4. Flujo completo del experimento

```text
Datos BTCUSDT y ETHUSDT
          |
          v
Velas conjuntas de 6 horas y retornos logarítmicos
          |
          v
Ventanas: 60 días de contexto -> 30 días de futuro
          |
          v
Split temporal con purgas y normalización de entrenamiento
          |
          +----------------+----------------+----------------+
          |                |                |                |
          v                v                v                v
     Bootstrap           CVAE          RealNVP             GAN
          |                |                |                |
          +----------------+----------------+----------------+
                                   |
                                   v
                     Evaluación común en test real
                                   |
                      +------------+-------------+
                      |                          |
                      v                          v
             Riesgo de cartera          Utilidad downstream
```

Para la evaluación principal, cada modelo genera 20 escenarios para cada una
de las 1.826 condiciones de prueba: **36.520 trayectorias sintéticas por
modelo**.

## 5. Los cuatro modelos

### 5.1 Block Bootstrap condicionado

#### Intuición

Es como buscar en el pasado momentos parecidos al mercado actual, recortar
fragmentos de esos periodos y unirlos para construir un nuevo futuro.

#### Cómo funciona

- Busca los 128 estados históricos de entrenamiento más cercanos a la
  condición actual.
- Elige bloques conjuntos de BTC y ETH de 12 pasos, equivalentes a 3 días.
- Mantiene BTC y ETH alineados dentro de cada bloque.
- Une bloques hasta completar los 120 pasos del horizonte futuro.

Es un método **no paramétrico**: no aprende una red neuronal ni supone una forma
matemática cerrada para toda la distribución.

#### Utilidad

Es el baseline o punto de referencia. Si una red compleja no supera este método
sencillo, su complejidad adicional puede no estar justificada.

#### Fortalezas observadas

- mejor fidelidad marginal: `0,1063`;
- menor error de correlación en estrés: `0,0384`;
- menor error de dependencia de cola: `0,0394`;
- estimación agregada de riesgo más conservadora.

#### Limitaciones

Solo reorganiza fragmentos históricos existentes. Además, la condición se fija
al comenzar la trayectoria: el método no va actualizando dinámicamente el
régimen mientras genera el futuro.

En downstream, validación selecciona `+25 %`, pero el MAE de test empeora de
`7,955 %` a `8,468 %`. Preservar bien la distribución no garantiza ayudar a una
predicción.

### 5.2 CVAE condicionado con distribución Student-t

Este es el modelo principal de la parte de Javi.

#### Intuición

El CVAE comprime los posibles futuros en un pequeño espacio de ideas o factores
ocultos, llamado **espacio latente**. Al muestrear distintos puntos de ese
espacio para el mismo contexto de mercado, obtenemos futuros diferentes pero
compatibles con la misma situación inicial.

Una analogía útil:

```text
Contexto actual + dado probabilístico -> un futuro plausible
Contexto actual + otro lanzamiento    -> otro futuro plausible
```

El dado no produce resultados arbitrarios: el contexto condiciona el tipo de
futuros que el decodificador puede generar.

#### Cómo aprende

Durante el entrenamiento tiene dos partes:

1. **Encoder:** observa el futuro real y las 14 condiciones, y lo resume en una
   distribución latente mediante una media y una varianza.
2. **Decoder:** recibe una muestra latente y las condiciones, y reconstruye una
   distribución para la trayectoria futura de BTC y ETH.

Se utiliza el truco de reparametrización para poder muestrear del espacio
latente y entrenar la red mediante descenso de gradiente.

La función de pérdida combina:

- el error probabilístico de reconstrucción, mediante la log-verosimilitud
  negativa de una Student-t;
- una penalización KL que ordena el espacio latente y permite muestrear nuevos
  futuros.

#### ¿Por qué una Student-t?

Los retornos financieros tienen más valores extremos que una distribución
normal. La Student-t tiene colas más gruesas y resulta más adecuada para
representar movimientos grandes. El decoder produce en cada paso:

- localización de BTC y ETH;
- escala de BTC y ETH;
- correlación entre ambos activos.

La configuración final usa un espacio latente de 8 dimensiones, `beta = 0,01`,
`free bits = 0,02` y 5 grados de libertad para la Student-t. Se escogió la
configuración `student_medium_l8_b01` después de comparar 39 candidatos y
comprobar su estabilidad.

#### Cómo genera nuevos escenarios

Al generar ya no necesitamos el encoder:

1. fijamos las 14 variables del contexto;
2. muestreamos un vector aleatorio de 8 dimensiones;
3. el decoder produce los parámetros Student-t de 120 pasos para BTC y ETH;
4. muestreamos los retornos y obtenemos una trayectoria conjunta.

#### Resultados principales

- entrenamiento detenido tras 18 épocas mediante early stopping de validación;
- el conjunto de test no interviene en la selección;
- discriminador externo: `74,5 %`, el mejor valor entre las redes;
- distancia de regímenes: `0,139`, la mejor de los cuatro modelos;
- W1 de volatilidad realizada: `0,316`, la mejor;
- cobertura de la referencia: `0,948`;
- con la mezcla `+25 %` elegida en validación, MAE de test `7,770 %` frente a
  `7,955 %` con datos reales: mejora relativa del `2,32 %`.

#### Qué significa realmente el 74,5 %

Entrenamos un clasificador externo para distinguir trayectorias reales de
sintéticas. Un `50 %` sería ideal: significaría que clasifica como si lanzara
una moneda. Un `100 %` sería muy malo: separación perfecta.

Por eso `74,5 %` no significa que el CVAE tenga un 74,5 % de calidad. Significa
que todavía se puede distinguir, aunque es el modelo neuronal más difícil de
separar de los datos reales.

#### Limitación importante

El CVAE no está suficientemente calibrado para funcionar por sí solo como
motor de riesgo. Su VaR 95 % presenta aproximadamente un `20-22 %` de
excepciones, cuando se esperaría alrededor de un `5 %`. Además, 20 escenarios
por condición son pocos para interpretar con precisión VaR y ES al 99 %.

### 5.3 Normalizing Flow condicionado, RealNVP

#### Intuición

El Flow aprende una transformación reversible entre una distribución sencilla
y una trayectoria financiera compleja:

```text
ruido sencillo z <-> trayectoria BTC-ETH x
```

Como la transformación es invertible, puede generar trayectorias y también
calcular una densidad exacta **dentro del modelo aprendido**.

#### Cómo funciona

- aplana los 120 pasos de los dos activos en 240 valores;
- emplea 8 capas de acoplamiento afín;
- cada transformación también recibe las 14 condiciones;
- se entrena minimizando la log-verosimilitud negativa o NLL.

La densidad exacta no es la probabilidad verdadera del mercado: es la
probabilidad asignada por el modelo entrenado.

#### Fortalezas observadas

- mejor persistencia de volatilidad: `0,0491`;
- mejor W1 del retorno final: `0,3332`;
- mejor W1 de máximo drawdown: `0,1616`;
- mejores errores de cobertura VaR 95 % y 99 %: `0,1263` y `0,1242`.

#### Limitaciones

El mejor checkpoint aparece en la época 5, con NLL de validación por dimensión
de `0,798`, y después la validación empeora: hay sobreajuste. Su discriminador
externo alcanza `97,8 %`, por lo que sus trayectorias son fáciles de separar de
las reales.

En downstream, validación mantiene la opción de datos únicamente reales. La
mejora de test observada con `+100 %` sintético es descriptiva y no una mezcla
seleccionada correctamente con validación.

### 5.4 Conditional GAN

#### Intuición

Dos redes compiten:

- el **generador** intenta producir trayectorias que parezcan reales;
- el **discriminador** intenta detectar cuáles son falsas.

Ambos reciben el contexto del mercado. El generador mejora intentando engañar
al discriminador y el discriminador mejora intentando no ser engañado.

#### Cómo funciona

- el generador recibe ruido de 64 dimensiones y las 14 condiciones;
- produce una trayectoria aplanada de 240 retornos;
- el discriminador recibe la trayectoria y las condiciones;
- se entrena con una pérdida adversarial de clasificación;
- se comparan 3 configuraciones y se selecciona `GAN_3` mediante validación.

El modelo final se entrena durante 80 épocas y conserva el checkpoint de la
época 70, sin usar test para elegirlo.

#### Resultados y contradicción aparente

- cobertura de la referencia: `0,950`, la mejor;
- discriminador externo: `100 %`, perfectamente separable y por tanto malo;
- error de correlación en estrés: `0,518`, el peor;
- con `+100 %` sintético obtiene MAE de test `7,052 %`, una mejora descriptiva
  del `11,4 %`.

Ese `7,052 %` no convierte a la GAN en ganadora. La mezcla `+100 %` no fue
elegida por validación: para la GAN, validación selecciona datos solo reales.
Además, todos los R² de test siguen siendo negativos.

La lección es importante: un generador puede aportar una señal útil a una
tarea concreta aunque sus datos no sean globalmente realistas.

## 6. ¿Cómo evaluamos los modelos?

No sumamos todo en una única nota. Separamos tres preguntas.

### 6.1 Fidelidad

Pregunta: **¿los datos sintéticos se parecen a los reales?**

Se comparan:

- distribuciones de retornos;
- autocorrelación y persistencia de la volatilidad;
- correlación BTC-ETH;
- dependencia en las colas;
- retorno acumulado;
- volatilidad realizada;
- máximo drawdown;
- diversidad y posible memorización;
- cobertura de distintos regímenes.

En las distancias y errores, un valor menor suele ser mejor.

### 6.2 Riesgo

Pregunta: **¿representan correctamente las pérdidas y los extremos?**

Aplicamos las trayectorias a una cartera buy-and-hold con `60 % BTC` y
`40 % ETH`, sin rebalanceo, costes, apalancamiento ni derivados.

- **VaR 99 %:** umbral de pérdida que solo debería superarse aproximadamente
  en el peor 1 % de los casos.
- **ES 99 %:** pérdida media dentro de ese peor 1 %.
- **Máximo drawdown:** mayor caída desde un máximo previo hasta un mínimo
  posterior. No es necesariamente la pérdida final de la trayectoria.

| Escenarios | VaR 99 % | ES 99 % |
|---|---:|---:|
| Histórico de prueba | 32,48 % | 33,52 % |
| Block Bootstrap | 46,42 % | 52,42 % |
| CVAE | 28,32 % | 33,69 % |
| Normalizing Flow | 30,14 % | 35,37 % |
| Conditional GAN | 35,89 % | 42,79 % |

El Bootstrap es el más conservador en la distribución agregada. Que el ES del
CVAE esté cerca del histórico no demuestra que esté bien calibrado para cada
condición particular.

### 6.3 Utilidad downstream

Pregunta: **¿los datos sintéticos ayudan en una tarea posterior?**

La tarea consiste en predecir la magnitud del máximo drawdown a 30 días de la
cartera 60/40. Usamos siempre la misma MLP:

```text
14 variables -> 64 neuronas -> 32 neuronas -> 1 predicción
```

Solo cambia la cantidad de datos sintéticos añadidos al entrenamiento:

| Etiqueta | Reales | Sintéticos añadidos | Total | Porcentaje sintético real del total |
|---|---:|---:|---:|---:|
| Solo reales | 4.710 | 0 | 4.710 | 0 % |
| +25 % | 4.710 | 1.178 | 5.888 | 20 % |
| +50 % | 4.710 | 2.355 | 7.065 | 33,3 % |
| +100 % | 4.710 | 4.710 | 9.420 | 50 % |

`+25 %` significa añadir una cantidad sintética equivalente al 25 % del train
real; no significa que el conjunto final sea 25 % sintético.

Validación y test siempre son 100 % reales. La mezcla debe escogerse por el MAE
de validación y solo después se consulta el resultado de test.

| Generador | Mezcla seleccionada con validación | MAE de test |
|---|---:|---:|
| Block Bootstrap | +25 % | 8,468 % |
| CVAE | +25 % | 7,770 % |
| Normalizing Flow | Solo reales | 7,955 % |
| Conditional GAN | Solo reales | 7,955 % |

El CVAE es el único cuya mejora seleccionada en validación se mantiene en test.
Aun así, todos los R² son negativos. La mejora es modesta y no convierte la MLP
en un predictor satisfactorio.

## 7. Resultado general: no hay un ganador global

| Modelo | Destaca en | Principal debilidad |
|---|---|---|
| Bootstrap | Dependencia conjunta, colas y prudencia | No mejora downstream en test |
| CVAE | Regímenes, volatilidad y equilibrio | Mala calibración condicional de VaR |
| Flow | Dinámica temporal, VaR y drawdown | Sobreajuste y fácil separación |
| GAN | Cobertura y señal downstream descriptiva | Mala fidelidad conjunta |

La palabra **mejor** necesita siempre una frase adicional: ¿mejor para qué?

- Para preservar fragmentos históricos conjuntos y ser conservadores, el
  Bootstrap es atractivo.
- Para obtener el modelo neuronal más equilibrado, destaca el CVAE.
- Para determinadas propiedades temporales y de riesgo, destaca el Flow.
- Para observar una señal downstream agresiva, la GAN es interesante, pero no
  es la elección validada.

## 8. Qué hace cada notebook

1. [`01_block_bootstrap_multivariante.ipynb`](notebooks/01_block_bootstrap_multivariante.ipynb):
   construye y evalúa el baseline condicionado.
2. [`CVAE_BTC_ETH.ipynb`](notebooks/CVAE_BTC_ETH.ipynb): arquitectura,
   entrenamiento, selección y evaluación del CVAE.
3. [`Normalizing_Flow_BTC_ETH.ipynb`](notebooks/Normalizing_Flow_BTC_ETH.ipynb):
   entrenamiento y evaluación del RealNVP condicionado.
4. [`GAN_BTC_ETH.ipynb`](notebooks/GAN_BTC_ETH.ipynb): búsqueda, entrenamiento
   y evaluación de la GAN condicionada.
5. [`02_aplicacion_cartera_y_stress.ipynb`](notebooks/02_aplicacion_cartera_y_stress.ipynb):
   aplica todos los escenarios a la cartera y calcula el riesgo.
6. [`03_generacion_masiva_ultimo_estado.ipynb`](notebooks/03_generacion_masiva_ultimo_estado.ipynb):
   genera 100.000 escenarios por modelo para la última condición disponible en
   el dataset. No significa que consulte el mercado en tiempo real.
7. [`05_comparacion_downstream_comun.ipynb`](notebooks/05_comparacion_downstream_comun.ipynb):
   entrena la misma MLP con todas las mezclas reales/sintéticas.
8. [`04_comparacion_final_consolidada.ipynb`](notebooks/04_comparacion_final_consolidada.ipynb):
   reúne los resultados utilizados en la comparación y en la presentación.

## 9. Parte de Javi: diapositiva 6

### Mensaje que debe recordar el público

> El CVAE genera múltiples futuros condicionados por el mercado reciente y es
> el modelo neuronal más equilibrado, aunque todavía no es perfecto.

### Cómo explicar los histogramas

- Azul representa datos reales y naranja datos generados por el CVAE.
- Los gráficos superiores muestran retornos individuales de 6 horas.
- Los inferiores muestran el retorno acumulado durante los 30 días.
- Cuanto más se solapan las formas, más parecidas son las distribuciones.
- Las diferencias laterales muestran que el CVAE suaviza o representa peor
  algunas zonas de la distribución, especialmente ciertos modos extremos.

No conviene decir que los histogramas son idénticos. El mensaje correcto es que
existe un solapamiento razonable, pero quedan diferencias visibles.

### Cómo explicar las cuatro cifras

- **74,5 % de discriminador externo:** es el mejor de las redes, pero `50 %`
  sería lo ideal. Cuanto menor, mejor.
- **0,139 de distancia entre regímenes:** el CVAE es el que mejor cubre los
  distintos tipos de mercado. Cuanto menor, mejor.
- **0,316 de W1 de volatilidad realizada:** reproduce mejor la distribución de
  volatilidad a lo largo de las trayectorias. Cuanto menor, mejor.
- **7,770 % de MAE:** al añadir `+25 %` de sintéticos, seleccionado mediante
  validación, mejora modestamente el `7,955 %` de solo datos reales.

### Entrenamiento

El entrenamiento termina en 18 épocas mediante early stopping sobre la pérdida
de validación. El test se reserva hasta el final. Esto evita elegir el modelo
que casualmente funciona mejor en el examen final.

## 10. Parte de Javi: diapositiva 7

### Cómo leer el mapa de calor

- Cada fila es una dimensión distinta de evaluación.
- Cada columna es un generador.
- El rango `1`, en verde, es el mejor de esa fila.
- El rango `4`, en rojo, es el peor.

No hay una columna completamente verde. Esa es la demostración visual de que no
existe un ganador universal.

### Cómo leer las curvas de la derecha

- El eje horizontal indica cuántos datos sintéticos se añaden.
- El gráfico izquierdo muestra el MAE global de test.
- El gráfico derecho muestra el MAE en el peor 10 % de los casos.
- En ambos, un valor más bajo es mejor.

Las curvas enseñan dos cosas:

1. añadir más datos sintéticos no siempre ayuda;
2. una mejora media puede no coincidir con una mejora en los casos extremos.

La GAN baja mucho el MAE de test con `+100 %`, pero esa mezcla no fue elegida
por validación. Por eso se presenta como señal descriptiva, no como victoria.

### Conclusión defendible

> Los generadores deben compararse separando fidelidad estadística,
> representación del riesgo y utilidad predictiva. Un único score ocultaría
> compromisos importantes entre esas dimensiones.

## 11. Guion oral recomendado para Javi

Esta versión conserva las ideas esenciales y deja algo más de margen para
respirar, mirar al público y señalar los gráficos.

### Diapositiva 6 - CVAE condicionado

> El CVAE genera múltiples futuros posibles mediante un espacio latente
> probabilístico condicionado por el contexto reciente del mercado.
>
> Durante el entrenamiento, el encoder resume las trayectorias reales en ese
> espacio latente. Después, el decoder combina una muestra aleatoria con las 14
> señales del contexto para generar 30 días conjuntos de Bitcoin y Ethereum.
>
> El modelo se detuvo tras 18 épocas usando únicamente la pérdida de validación;
> el test no intervino en la selección.
>
> Es el modelo neuronal más equilibrado: obtiene el mejor resultado entre las
> redes frente al discriminador externo y lidera en cobertura de regímenes y
> volatilidad realizada. Además, la mezcla de un 25 % adicional de datos
> sintéticos reduce el MAE de test del 7,955 al 7,770 %.
>
> No domina todas las métricas, pero sus escenarios son los más difíciles de
> separar de los reales entre los modelos neuronales.

### Diapositiva 7 - Conclusión

> La conclusión es que no existe un ganador global.
>
> El Bootstrap destaca preservando dependencia y colas; el CVAE, en regímenes
> y equilibrio general; el Flow, en varias propiedades temporales y de riesgo;
> y la GAN muestra una señal downstream fuerte, aunque no validada como mezcla
> final y con problemas claros de fidelidad conjunta.
>
> El mapa de calor muestra que el primer puesto cambia según la métrica. Las
> curvas de la derecha muestran además que añadir más datos sintéticos no
> siempre mejora la predicción y que el resultado medio puede ser diferente en
> los peores casos.
>
> Por eso debemos separar tres dimensiones: fidelidad estadística,
> representación del riesgo y utilidad predictiva. Resumirlas en una sola nota
> ocultaría compromisos relevantes.
>
> En definitiva, los datos sintéticos pueden aportar valor, pero ese valor debe
> demostrarse mediante validación, no asumirse porque los escenarios parezcan
> realistas. Muchas gracias.

## 12. Preguntas probables y respuestas cortas

### ¿Estáis prediciendo el precio de BTC y ETH?

No. Generamos distribuciones de posibles trayectorias de retornos, no una única
predicción puntual del precio.

### ¿Qué aporta que los modelos sean condicionados?

Permite que la distribución futura dependa del estado reciente del mercado. Un
contexto tranquilo y uno de crisis no deberían producir los mismos futuros.

### ¿Por qué el CVAE es vuestro modelo neuronal más equilibrado?

Porque combina el mejor discriminador entre las redes, la mejor cobertura de
regímenes y de volatilidad, y una pequeña mejora downstream seleccionada por
validación. No gana todo, pero evita debilidades tan extremas como las de otros
modelos.

### ¿Un discriminador de 74,5 % es bueno?

Es el mejor de los modelos neuronales del proyecto, pero no es ideal. El valor
perfecto sería aproximadamente 50 %, porque significaría que no puede distinguir
los datos reales de los sintéticos.

### ¿Por qué no gana la GAN si obtiene 7,052 % de MAE?

Porque ese resultado corresponde a una mezcla `+100 %` que no fue elegida por
validación. Además, su discriminador externo llega al 100 % y reproduce mal la
correlación en estrés. Es un resultado descriptivo, no una elección final
justificada.

### ¿Por qué no usar solo el Bootstrap si gana varias métricas?

Porque gana algunas dimensiones de fidelidad y es conservador, pero no mejora
la tarea downstream en test. El mejor método depende del uso financiero.

### ¿Los 100.000 escenarios añaden información histórica nueva?

No. Aumentan la resolución de la distribución aprendida para una condición,
pero toda la información procede de los datos y supuestos usados al entrenar.

### ¿Podría utilizarse ya el CVAE como motor de riesgo real?

No sin más validación y mejoras. Su calibración condicional de VaR es pobre, las
ventanas se solapan y el estudio no incluye todos los riesgos y costes de un
sistema financiero real.

### ¿Cuál es la aportación principal del trabajo?

Un protocolo común y honesto para comparar cuatro generadores en tres planos:
fidelidad, riesgo y utilidad. El resultado central no es coronar un modelo, sino
mostrar que esas tres cualidades no son equivalentes.

## 13. Errores que conviene evitar al presentar

- No decir que Flow aprende la probabilidad verdadera del mercado; aprende una
  densidad exacta bajo su propio modelo.
- No decir que un discriminador alto es mejor. En esta métrica, `50 %` es el
  ideal y `100 %` es malo.
- No decir que la GAN gana por obtener `7,052 %`; ese resultado es descriptivo
  y no fue seleccionado con validación.
- No decir que el CVAE está bien calibrado para VaR; su tasa de excepciones es
  demasiado alta.
- No llamar USD fiat a USDT.
- No tratar las 9.096 ventanas como episodios independientes.
- No confundir `+25 %` sintético con un conjunto final que contiene 25 % de
  sintéticos: en realidad contiene aproximadamente un 20 %.
- No decir que el Bootstrap actual es incondicionado: utiliza las mismas 14
  variables de contexto que los modelos neuronales.

## 14. Tres frases para memorizar

Si solo recuerdas tres ideas, que sean estas:

1. **El CVAE genera múltiples futuros BTC-ETH condicionados por el mercado
   reciente mediante un espacio latente probabilístico.**
2. **Es el modelo neuronal más equilibrado y consigue una mejora downstream
   modesta, elegida correctamente con validación.**
3. **No existe un ganador global: fidelidad, riesgo y utilidad predictiva deben
   evaluarse por separado.**

## 15. Material de referencia

- [Presentación revisada](Presentacion_Taller_Generativos_BTC_ETH_revisada.pdf)
- [Guion cronometrado](Guion_presentacion_BTC_ETH_5_minutos.pdf)
- [Enunciado del taller](Taller_B5_T1.pdf)
- [README del proyecto](README.md)
- [Procedencia y preparación de datos](data/README.md)
- [Resultados downstream comunes](outputs/downstream_common/comparison.csv)

## 16. Mi propuesta final para la intervención de Javi

Esta sería mi versión definitiva partiendo de la sección de Javier en el
[`Guion_presentacion_BTC_ETH_5_minutos.pdf`](Guion_presentacion_BTC_ETH_5_minutos.pdf).
Añade una explicación breve de la arquitectura sin convertir la exposición en
una lista técnica de capas.

### Diapositiva 6 - CVAE condicionado · 3:05-4:00

**[Empezar mirando al público]**

> El CVAE genera múltiples futuros condicionados por el contexto reciente. Su
> encoder procesa la trayectoria real de 120 pasos con dos capas
> convolucionales y las 14 señales con una rama densa. Después combina ambas y
> aprende una distribución latente de 8 dimensiones mediante una media y una
> varianza.

**[Hacer con las manos un gesto de comprimir y después de desplegar]**

> El decoder toma una muestra de ese espacio, reincorpora el contexto y genera
> 120 pasos. En cada uno produce la localización y escala de ambos activos y su
> correlación mediante una Student-t, adecuada para las colas financieras.

**[Señalar los histogramas]**

> Tras 18 épocas, vemos un solapamiento razonable, aunque no perfecto. Es la red
> más difícil de distinguir de los datos reales y lidera en regímenes y
> volatilidad realizada.

**[Señalar 7,770 %]**

> Añadir un 25 % de sintéticos, elegido mediante validación, reduce el MAE de
> test del 7,955 al 7,770 %. Por eso es la red más equilibrada.

### Diapositiva 7 - Conclusión · 4:00-5:00

**[Señalar el mapa de calor]**

> No existe un ganador global. El Bootstrap destaca en dependencia y colas; el
> CVAE, en regímenes y equilibrio; el Flow, en dinámica y riesgo; y la GAN
> aporta una señal downstream llamativa, pero con problemas de fidelidad. El
> mapa de calor muestra cómo cambia el ganador según la dimensión.

**[Señalar las curvas de la derecha]**

> Las curvas muestran que añadir más sintéticos no siempre ayuda y que el
> resultado medio puede diferir en los peores casos. El modelo adecuado depende
> del objetivo financiero.

**[Dejar de mirar la pantalla, bajar el ritmo y cerrar mirando al público]**

> Debemos separar fidelidad, representación del riesgo y utilidad predictiva.
> Una única puntuación ocultaría compromisos importantes.
>
> Los datos sintéticos pueden aportar valor, pero debemos demostrarlo mediante
> validación, no asumirlo porque parezcan realistas. Muchas gracias.

### Arquitectura del CVAE en una imagen mental

Si te preguntan por la arquitectura, puedes dibujarla verbalmente así:

```text
Trayectoria real [120 x 2]
        |
        v
Conv1D 32 -> Conv1D 64 -> aplanado ----+
                                          |
14 condiciones -> Dense 24 -> Dense 48 --+-> Dense 96
                                                  |
                                      media y log-varianza
                                                  |
                                     espacio latente z [8]
                                                  |
14 condiciones -> Dense 24 -> Dense 48 ----------+
                                                  |
                                                  v
                                  Dense -> reshape [30 x 64]
                                                  |
                              ampliar x2 + Conv1D 64
                                                  |
                              ampliar x2 + Conv1D 32
                                                  |
                                                  v
                          120 pasos x 5 parámetros Student-t
                         loc BTC/ETH, escala BTC/ETH y correlación
```

La frase corta para defenderla es:

> El encoder comprime el futuro real y su contexto en una distribución latente
> de 8 dimensiones; el decoder toma una muestra de esa distribución, vuelve a
> incorporar el contexto y genera una distribución Student-t bivariante para
> cada uno de los 120 pasos futuros.
