# Laboratorio 01 — Deep Learning: clasificación ordinal de GDS

**Estudiantes:** Gael Ortega y Johan Piñones

**Asignatura:** Deep Learning · segundo semestre de 2026

**Estado:** estructura de trabajo; resultados y conclusiones pendientes de congelar

## 1. Resumen y objetivos

Este laboratorio compara una red Softmax con una salida ordinal CORAL para
predecir el deterioro cognitivo a partir de 15 atributos binarios. Cada columna
`GDS`, `GDS_R1`, ..., `GDS_R5` constituye un experimento independiente.

**Completar al final:** pregunta experimental, métodos efectivamente incluidos
y resumen de los hallazgos. El informe se redactará después en Overleaf a partir
de este README y de las tablas definitivas.

## 2. Dataset y variables objetivo

El archivo local `dataset/15 atributos R0-R5.sav` contiene 1119 observaciones,
15 entradas binarias y seis objetivos. El SAV no se versiona. Cada objetivo se
recodifica a índices consecutivos `0, ..., K-1` para entrenar, manteniendo la
correspondencia con sus etiquetas originales para interpretar los resultados.

| Objetivo | Clases |
| --- | ---: |
| `GDS` | 7 |
| `GDS_R1` | 3 |
| `GDS_R2` | 3 |
| `GDS_R3` | 2 |
| `GDS_R4` | 3 |
| `GDS_R5` | 3 |

**Completar al final:** significado confirmado de los valores 0/1, tabla de
recodificaciones, frecuencias de clase y análisis de patrones ambiguos.

## 3. Estructura del proyecto

| Ruta | Función |
| --- | --- |
| `main.py` | Ejecuta los experimentos y guarda los resultados. |
| `src/config.py` | Define objetivos, valores por defecto y grid. |
| `src/data_loader.py`, `src/preprocessing.py` | Cargan datos, recodifican objetivos y crean particiones. |
| `src/models.py`, `src/losses.py`, `src/ordinal.py`, `src/methods.py` | Definen redes, pérdidas y predicciones. |
| `src/evaluation.py`, `src/reporting.py` | Calculan métricas y generan reportes. |
| `tests/` | Comprueba contratos de datos, modelos y validación. |
| `results/` | Guarda corridas locales; su contenido se ignora en Git. |

**Completar al final:** ubicación versionada de las tablas y figuras escogidas.

## 4. Entorno y ejecución

Crear y activar el entorno definido en `environment.yml`:

```bash
conda env create -f environment.yml
conda activate lab_pytorch
```

Ejecutar una prueba corta y, luego, los seis objetivos:

```bash
python main.py --data-path 'dataset/15 atributos R0-R5.sav' --target-name GDS_R2 --outer-folds 2 --inner-folds 2 --epochs 1
python main.py --data-path 'dataset/15 atributos R0-R5.sav' --all-targets
```

Sin `--output-dir`, cada ejecución crea una carpeta con fecha y hora dentro de
`results/`. Los archivos `estado.json` indican si la corrida terminó o falló.

**Completar al final:** comando exacto de cada corrida elegida, versión del
código, versiones del entorno, dispositivo y semilla.

## 5. Métodos y diseño experimental

Las cuatro filas obligatorias son **Softmax fijo**, **Softmax HP**, **CORAL sin
pesos** y **CORAL con pesos por número efectivo de muestras**. Softmax entrega
`K` logits y CORAL entrega `K-1` logits de umbral. La pérdida de Softmax es
entropía cruzada; CORAL usa BCE con logits sobre etiquetas acumulativas.

La corrida H4 usa cinco folds externos y tres internos en `GDS_R1`–`GDS_R5`.
En `GDS` usa dos externos y dos internos; para estratificar internamente agrupa
las etapas originales 6 y 7, pero entrena y evalúa con las siete etiquetas.
Softmax HP selecciona entre cuatro configuraciones por menor MAE interno y,
en empate, mayor QWK. Los otros tres métodos usan configuración fija en H4;
CORAL con pesos usa β=0,99.

**Pendiente de implementación H5:** el plan propone buscar cuatro
configuraciones para CORAL y CORN, y cruzarlas con β en `{0.9, 0.99, 0.999}`
para CORAL con pesos. Actualizar esta sección solo después de comprobar la
corrida definitiva. Documentar la ausencia inevitable de la etapa 7 en uno de
los entrenamientos internos de `GDS`. Describir CORN y otros controles solo si
se implementan, comprueban y se incluyen en las corridas definitivas.

## 6. Métricas

Se reportan media y desviación de los folds externos para `accuracy`,
`balanced_accuracy`, precisión, recall y F1 macro, MAE ordinal, QWK, exactitud
con tolerancia ±1 y proporción de errores de dos o más clases.

**Completar al final:** cómo se interpretan las métricas frente al desbalance
y por qué las comparaciones principales se hacen dentro de cada objetivo.

## 7. Resultados y análisis

**Completar después de congelar las corridas:** insertar una tabla por objetivo
con las cuatro filas obligatorias y las nueve métricas (media ± desviación
externa). Añadir solo las extensiones realmente ejecutadas.

### GDS

Pendiente: tabla, matriz de confusión y comentario sobre la clase extrema.

### GDS_R1

Pendiente: tabla y análisis por clase.

### GDS_R2

Pendiente: tabla y comparación de los cuatro métodos.

### GDS_R3

Pendiente: tabla y comentario sobre el objetivo binario.

### GDS_R4

Pendiente: tabla y efecto del desbalance.

### GDS_R5

Pendiente: tabla y análisis por clase.

## 8. Reproducibilidad, limitaciones y conclusiones

**Completar al final:** referencias a las tablas y figuras versionadas,
configuraciones seleccionadas por fold, cobertura OOF, limitaciones del
protocolo y conclusiones basadas en las corridas congeladas.

Para ejecutar las pruebas actuales:

```bash
python -m unittest discover -s tests -v
```
