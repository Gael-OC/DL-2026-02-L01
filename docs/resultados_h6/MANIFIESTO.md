# Resultados definitivos H6

La corrida seleccionada para los seis objetivos y los seis entrenamientos por
objetivo es `results/h6_20260923_seed42`. La carpeta se nombró en el protocolo
H5 para el 23/09, pero la ejecución efectiva terminó el **22/09/2026 a las
22:59:22 -03** en el servidor `enterprise`. `results/` contiene los artefactos
completos y sigue ignorado por Git. Esta carpeta versiona las tablas y las
matrices de confusión OOF agregadas de esa misma corrida, sin mezclar H4 ni los
smoke tests de H5.

## Código, datos y entorno

- Commit del código copiado al servidor: `4d0b6e59b3cedc6828d5007923c04d8288a187ad`.
- SHA-256 de `main.py` en ambos equipos: `cba37a44b495d84346431631c5699073da7ce28ddf4aa071d34d259846f6d616`.
- SHA-256 del SAV en ambos equipos: `b454a4e5c997be26edb3a187605c2c10da1147ac1ce6cea76d42adbe68b17474`.
- Servidor: AMD EPYC 7543, dos NVIDIA RTX A5000; entorno `lab_pytorch`,
  Python 3.10.21, PyTorch 2.14.0+cu130, NumPy 2.2.6, pandas 2.3.3,
  scikit-learn 1.7.2, pyreadstat 1.3.6 y Matplotlib 3.10.9.
- Benchmark en `GDS_R2`, un fold de 559/560 observaciones, 20 épocas, batch
  32, Softmax fijo y seed 42; tres ajustes consecutivos por dispositivo:
  CPU 6.230/0.223/0.221 s, CUDA 0.578/0.428/0.428 s. El primer ajuste de
  CPU incluyó la inicialización; la mediana fue **0.223 s en CPU** frente a
  **0.428 s en CUDA**. Se escogió CPU para los muchos ajustes de H6.
- Las 29 pruebas pasaron en el servidor antes de la corrida.

Comando efectivo, ejecutado desde el directorio del proyecto copiado en el
servidor (`~/.envs/lab_pytorch/bin/python` es el Python del entorno activo):

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg ~/.envs/lab_pytorch/bin/python main.py --data-path 'dataset/15 atributos R0-R5.sav' --all-targets --methods softmax_fixed softmax_hp coral coral_weighted corn softmax_matched --outer-folds 5 --inner-folds 3 --gds-outer-folds 2 --gds-inner-folds 2 --epochs 20 --batch-size 32 --seed 42 --device cpu --output-dir results/h6_20260923_seed42
```

Se ejecutaron 5 folds externos × 3 internos para `GDS_R1`–`GDS_R5` y 2 × 2
para `GDS`. En sus folds internos, `GDS` agrupó las clases originales 6 y 7
solo para estratificar; el entrenamiento y la evaluación mantuvieron las siete
clases. El grid y la regla MAE/QWK quedaron fijados en H5 antes de esta
corrida. Softmax equiparado produjo argmax y mediana desde las mismas
probabilidades y checkpoints.

## Comprobaciones

`estado.json` terminó en `complete`, sin fallos, y las 42 combinaciones
método/objetivo tienen estado completo. Se verificaron 42 filas de resultados,
189 configuraciones de folds y nueve medias y desviaciones finitas por fila.
Cada fila tiene los 1119 índices originales OOF exactamente una vez: 46 998
predicciones en total. Todas las probabilidades son finitas, están en [0, 1]
y suman uno por muestra. Las 42 matrices OOF coinciden con sus predicciones;
argmax y mediana tienen probabilidades idénticas y el mismo fold externo.

Archivos finales:

- `resultados.csv` y `resultados.md`: nueve métricas como media ± desviación
  entre folds externos para cada método y objetivo.
- `configuraciones_folds.csv`: configuración elegida y puntuaciones internas
  por fold; Softmax fijo conserva la configuración base.
- `verificacion_oof.csv`: cobertura por método y objetivo.
- `matrices/<método>/<objetivo>.csv` y `.png`: confusiones OOF agregadas.

## Lectura dentro de cada objetivo

- `GDS`: CORAL sin pesos obtuvo F1 macro 0.366 frente a 0.222 de Softmax HP,
  pero MAE 0.709 frente a 0.509. Softmax equiparado con mediana dio MAE 0.477
  y QWK 0.697. Solo hay dos casos de clase 7; su recall es inestable.
- `GDS_R1`: la mediana equiparada alcanzó F1 macro 0.731 y CORN 0.721,
  frente a 0.505 de Softmax HP. CORAL con pesos subió a 0.629 desde 0.609
  sin pesos, con MAE 0.121 frente a 0.124.
- `GDS_R2`: la mediana equiparada dio F1 macro 0.677 y MAE 0.283; CORN
  0.668 y 0.289; Softmax HP 0.653 y 0.293. Los pesos de CORAL apenas
  cambiaron su F1 (0.637 a 0.636).
- `GDS_R3`: Softmax fijo y HP empataron en F1 macro 0.804 y MAE 0.090;
  CORAL obtuvo 0.789 y 0.097. Al ser binario, no existen errores de dos o
  más clases.
- `GDS_R4`: CORAL sin pesos mejoró F1 macro (0.600 frente a 0.513 de
  Softmax HP) y el recall de la clase 1 (0.537 frente a 0), pero elevó el
  MAE (0.300 frente a 0.168). Los pesos redujeron levemente el F1 a 0.588.
- `GDS_R5`: CORAL sin pesos obtuvo F1 macro 0.589 frente a 0.510 de
  Softmax HP y recuperó la clase 1 (recall 0.497 frente a 0), pero MAE
  subió de 0.222 a 0.346. CORAL con pesos dio las mismas medias redondeadas.

Estas son comparaciones descriptivas dentro de cada objetivo. El contraste
Softmax HP frente a Softmax equiparado informa sobre el cambio de
arquitectura; Softmax equiparado frente a CORAL y CORN informa sobre las
cabezas, pérdidas y decodificadores con un bloque oculto compartido. Las
configuraciones se seleccionaron por separado en cada fold, por lo que esos
contrastes no estiman un efecto causal aislado. CORAL con/sin pesos muestra el
efecto conjunto de ponderar y volver a seleccionar hiperparámetros. Argmax y
mediana sí usan exactamente el mismo ajuste y aíslan la regla de decisión.

El fuerte desbalance afecta F1 macro y balanced accuracy: `GDS_R1` tiene
947/150/22 casos; `GDS_R4`, 149/906/64. Los 15 atributos admiten perfiles
repetidos con etiquetas distintas: 97 perfiles que cubren 851 filas para
`GDS`, y entre 30 y 73 perfiles para cada objetivo recodificado. Esto limita
la exactitud posible a partir de esas entradas y exige cautela al interpretar
errores por clase. Es una sola semilla y no se calculó incertidumbre entre
semillas. No se eligieron métodos ni configuraciones mirando el test externo.
