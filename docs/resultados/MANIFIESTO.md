# Resultados del laboratorio

Estas tablas y figuras proceden de la comparación completa ejecutada el
23/09/2026 en `enterprise`. Los artefactos por fold y las predicciones OOF
permanecen en `results/`, que está ignorada por Git. Aquí se publican las
tablas, configuraciones por fold, cobertura OOF y 42 matrices agregadas en
CSV y PNG.

## Procedencia

- Código ejecutado: SHA-256 combinado de los 11 archivos Python del
  manifiesto de la corrida
  `8d859c5122b537d51d4e8d8382941ad57cf90c89cfa3e01150a6004d1915be20`.
  La copia del servidor no tenía `.git`, por lo que el manifiesto guarda
  hashes por archivo y no atribuye la ejecución al commit local.
- SAV: SHA-256
  `b454a4e5c997be26edb3a187605c2c10da1147ac1ce6cea76d42adbe68b17474`.
- Entorno: Python 3.10.21, PyTorch 2.14.0, NumPy 2.2.6, pandas 2.3.3,
  scikit-learn 1.7.2, pyreadstat 1.3.6; CPU con `OMP_NUM_THREADS=1` y
  `MPLBACKEND=Agg`.
- Protocolo: seed 42, batch 32, 20 épocas, 5 folds externos y 3 internos
  para `GDS_R1`–`GDS_R5`, 2 y 2 para `GDS`. Sus clases 6/7 se agrupan solo
  para estratificar internamente. El grid original tiene cuatro candidatos;
  CORAL ponderado cruza esos cuatro con β=0,9/0,99/0,999. La elección usa
  MAE interno medio y desempata por QWK interno medio.

Comando de reproducción desde la raíz del proyecto, con una carpeta de
salida nueva:

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg python main.py --data-path 'dataset/15 atributos R0-R5.sav' --all-targets --methods softmax_fixed softmax_hp coral coral_weighted corn softmax_matched --outer-folds 5 --inner-folds 3 --gds-outer-folds 2 --gds-inner-folds 2 --epochs 20 --batch-size 32 --seed 42 --device cpu --output-dir results/corrida_final
```

Después de ejecutar se abrevió únicamente el nombre visible del método
`corn` a «CORN» en `src/methods.py` y en las tablas públicas. No cambiaron
la pérdida, arquitectura, selección ni predicciones. Por ello, el hash del
código ejecutado identifica los archivos anteriores a ese ajuste de
presentación; las cifras de las tablas son las de la corrida validada.

## Verificación

`estado.json` indica `complete`, 42 experimentos y cero fallos. Se comprobaron
42 filas, 189 registros por fold, 46 998 predicciones OOF, los 1119 índices
exactamente una vez por combinación método/objetivo, probabilidades válidas,
cuatro o doce candidatos internos según el método, ganadores MAE/QWK, las
nueve métricas y sus medias/desviaciones, y las matrices de confusión. El
presupuesto reconstruido es de 2374 ajustes: 2212 internos y 162 finales.

La transferencia local contiene 898 archivos; el hash agregado de sus
contenidos y rutas relativas coincidió exactamente con el del servidor:
`9735798620ce6a08f832760f7c2daa743ce3db057a64818666e2fd48c70c805a`.
La normalización del nombre visible a «CORN» en los CSV y Markdown publicados
no altera las métricas ni configuraciones. Las matrices CSV y PNG se copiaron
sin modificar sus contenidos.

Las cifras de las tablas son medias ± desviaciones entre folds externos;
no son intervalos de confianza. Los CSV y PNG de confusión de esta carpeta
usan las **clases originales** en ambos ejes. `GDS_R3` usa internamente
índices 0/1 para sus clases originales 1/3. CORN usa la reducción de BCE por
el número total de pares muestra–umbral elegibles.
