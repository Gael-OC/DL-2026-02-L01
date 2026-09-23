# Plan de ejecución — Laboratorio 01

> Documento temporal de coordinación del equipo. Se mantiene durante el
> desarrollo y se elimina de la versión final solamente después de traspasar
> al `README.md` el procedimiento, los comandos, las decisiones y los
> resultados definitivos.

**Equipo:** Gael Ortega y Johan Piñones

**Entrega:** 24/09/2026 a las 23:59

**Calendario desde H4:** 22/09 cerrar el código y probarlo; 23/09 ejecutar
experimentos definitivos en el servidor, congelar resultados y completar el
README; 24/09 redactar el informe en Overleaf y efectuar la entrega.

La distribución de tareas permite avanzar en paralelo, reducir conflictos de
código y reunir el trabajo en puntos de integración acordados.

## 1. Objetivo del trabajo

Construir una comparación reproducible de métodos de clasificación nominal y
ordinal para los seis objetivos `GDS`, `GDS_R1`, ..., `GDS_R5`.

La entrega obligatoria debe incluir, como mínimo:

1. Softmax con la configuración entregada.
2. Softmax con búsqueda interna de hiperparámetros.
3. CORAL sin ponderación.
4. CORAL con pesos por número efectivo de muestras.
5. Validación anidada, métricas, tablas y trazabilidad del procedimiento.

Con el núcleo experimental obligatorio funcionando y tiempo reservado para
documentar la entrega, el objetivo del 22/09 es integrar **CORN** para la
corrida del 23/09. Softmax con arquitectura equiparada y la decisión por
mediana son controles adicionales. Solo entran en el informe si pasan el
smoke test y la corrida definitiva.

## 2. Forma de trabajo

- Hasta H4, Gael llevó validación experimental y flujo común; Johan trabajó en
  CORAL, ponderación y reportes.
- Desde el 22/09 las tareas pendientes quedan abiertas a cualquiera del
  equipo. El PR #5 no bloquea el cierre del código ni las corridas.
- Los cambios llegan a `main` mediante ramas cortas y pull requests.
- Un cambio se incorpora cuando tiene una comprobación reproducible y su
  efecto sobre el experimento queda documentado.
- El test externo nunca se usa para escoger configuración, época, método o
  decodificador.
- Los seis objetivos son experimentos independientes; no son una red con seis
  salidas.
- Ahora se deja un esqueleto del `README.md`. Se completa con los resultados
  definitivos después de congelar las corridas; luego se redacta el informe en
  Overleaf tomando ese README como base. Durante el desarrollo se guarda
  evidencia breve y verificable.

## 3. Registro de trabajo hasta H4

Las asignaciones siguientes describen el trabajo previo. H5–H9 contienen las
tareas pendientes sin responsable preasignado.

### Gael — validación experimental y flujo común hasta H4

Archivos que llevó hasta H4:

- `main.py`
- `src/config.py`
- `src/preprocessing.py`
- `src/evaluation.py`

Tareas:

- validar el contrato de datos y las seis recodificaciones;
- generar y conservar folds reproducibles;
- asegurar que no exista fuga del test externo;
- corregir métricas para conservar la escala completa de clases;
- implementar la búsqueda interna por menor MAE y desempate por mayor QWK;
- construir el motor común de entrenamiento y evaluación;
- definir el contrato común de los métodos;
- conectar al flujo común los cuatro métodos obligatorios.

### Johan — CORAL, ponderación, salidas y reportes hasta H4

Archivos que llevó hasta H4:

- `src/models.py`
- `src/losses.py`
- `src/ordinal.py`
- `src/reporting.py`
- pruebas unitarias de CORAL y reportes

Tareas:

- implementar `CoralLayer` y `MLPCoral`;
- implementar la codificación acumulativa de etiquetas;
- implementar la pérdida CORAL con logits;
- implementar pesos por número efectivo de muestras;
- implementar predicción y probabilidades de clase para CORAL;
- comprobar formas, monotonía, pesos y gradientes con ejemplos pequeños;
- exportar predicciones OOF, configuraciones elegidas, tablas y matrices de
  confusión usando el formato compartido;
- ejecutar comandos acordados y registrar sus artefactos.

### Regla para cambios fuera del área propia

La asignación anterior deja de restringir el trabajo a partir de H5. Se
conserva la interfaz del contrato común y se evita editar el mismo archivo en
paralelo. Todo cambio nuevo debe partir del `main` actual y revisarse por
separado antes de incluirlo.

## 4. Contrato común de los métodos

El motor experimental debe poder seleccionar el método sin duplicar los ciclos
de entrenamiento y evaluación.

```text
build_model(method, config, num_features, num_classes) -> model
loss(method, logits, labels, context)                  -> escalar
decode(method, logits)                                -> enteros (B,)
predict_proba(method, logits)                          -> probabilidades (B,K)
```

Invariantes:

- `model(X)` recibe `X` de forma `(B, 15)` y devuelve logits.
- La pérdida devuelve un escalar finito y diferenciable.
- `decode` devuelve clases enteras entre `0` y `K-1`.
- `predict_proba` devuelve forma `(B, K)`, valores finitos y filas que suman
  aproximadamente uno.
- Softmax produce `K` logits; CORAL y CORN producen `K-1`.
- Cada ajuste crea un modelo y optimizador nuevos.

## 5. Prioridades

### P0 — entrega obligatoria

- Cuatro métodos obligatorios funcionando en los seis objetivos.
- Nested CV real para la búsqueda de hiperparámetros.
- Selección por MAE interno y desempate por QWK interno.
- Excepción de `GDS` acordada y documentada.
- Nueve métricas, configuraciones seleccionadas y predicciones OOF.
- Comandos y resultados reproducibles.
- Tablas finales accesibles, README completo e informe PDF coherente con el
  código entregado. H4 solo cierra la parte experimental de P0.

### P1 — laboratorio sobresaliente

- CORN con la misma base oculta usada en la comparación ordinal.
- Softmax con arquitectura equiparada.
- Argmax frente a mediana sobre las mismas probabilidades Softmax.
- Análisis de patrones de entrada ambiguos, desbalance y errores por clase.

### P2 — solo si P0 y P1 están cerrados

- Semillas adicionales en objetivos seleccionados.
- Ablación con hiperparámetros fijos.

No se implementará SORD ni otra familia adicional antes de cerrar P0 y P1.

## 6. Hitos y checklist

La casilla se marca únicamente cuando existe evidencia: test, comando,
archivo generado o pull request integrado.

### H0 — protocolo y entorno

- [x] Confirmar fecha y hora exactas de entrega.
- [x] Confirmar entorno de Python/PyTorch y dispositivo de ejecución.
- [ ] Ambos pueden ejecutar un smoke test de Softmax en `GDS_R2`.
- [x] Congelar contrato de métodos y formato mínimo de resultados.
- [x] Acordar la validación interna especial de `GDS` agrupando las clases 6 y 7 solo para estratificar.
- [x] Registrar versiones, seed inicial y comando de smoke test.

**Cierre:** ambos pueden iniciar sus módulos sin esperar código del otro.

### H1 — base experimental y CORAL en paralelo

#### Gael

- [x] Verificar columnas, tipos, valores finitos y recodificaciones.
- [x] Verificar conteos de clase de los seis objetivos.
- [x] Persistir o reconstruir determinísticamente los folds compartidos.
- [x] Probar disjunción entre outer-train y outer-test.
- [x] Probar que outer-test no aparece en los folds internos.
- [x] Corregir métricas y confusiones para usar la escala completa.

#### Johan

- [x] Implementar `labels_to_levels` para `K=2`, `K=3` y `K=7`.
- [x] Implementar `CoralLayer` con umbrales ordenados.
- [x] Implementar `MLPCoral`.
- [x] Implementar la pérdida CORAL sin aplicar sigmoid previamente.
- [x] Implementar `logits_to_ordinal_predictions`.
- [x] Añadir pruebas de formas, finitud y gradientes.

**Cierre:** la base Softmax conserva folds correctos y CORAL supera pruebas
sintéticas sin depender todavía de `main.py`.

### H2 — búsqueda interna y ponderación

#### Gael

- [x] Aplicar en `GDS` la agrupación de las clases originales 6 y 7 solo
      para estratificar los folds internos, conservando las etiquetas
      originales para entrenamiento y evaluación.
- [x] Recorrer las cuatro configuraciones de `HYPERPARAMETER_GRID`.
- [x] Promediar MAE y QWK sobre todos los folds internos.
- [x] Elegir menor MAE; en empate, mayor QWK.
- [x] Reentrenar desde cero sobre todo outer-train.
- [x] Guardar la mejor configuración de cada outer fold.
- [x] Verificar con un caso controlado que outer-test no influye en la selección.

#### Johan

- [x] Implementar pesos por número efectivo y normalizarlos a media uno.
- [x] Manejar clases ausentes sin divisiones por cero ni `NaN`.
- [x] Aplicar el peso de la clase de cada muestra a la pérdida CORAL.
- [x] Comprobar que pesos iguales a uno reproducen la pérdida no ponderada.
- [ ] Preparar un ejemplo numérico breve de CORAL para la documentación final
      (tarea abierta en H8).

**Cierre:** Softmax HP funciona correctamente y CORAL ponderado pasa sus
pruebas unitarias.

### H3 — integración P0 en `GDS_R2`

- [x] Integrar Softmax fijo mediante el contrato común.
- [x] Integrar Softmax HP mediante el contrato común.
- [x] Integrar CORAL sin pesos.
- [x] Integrar CORAL con pesos.
- [x] Producir cuatro filas comparables para `GDS_R2`.
- [x] Guardar la configuración elegida por outer fold.
- [x] Guardar predicciones OOF y probabilidades.
- [x] Revisar manualmente una matriz de confusión y una corrida.

**Evidencia H3 (smoke test, no resultados finales):**
`python main.py --data-path 'dataset/15 atributos R0-R5.sav' --target-name GDS_R2 --outer-folds 2 --inner-folds 2 --epochs 1 --output-dir results/h3_gds_r2_smoke`.
Generó cuatro filas, `configuraciones_folds.csv`, `predicciones_oof.csv` y
cuatro matrices de confusión. Cada método tiene 1119 índices OOF distintos
(0..1118), una vez cada uno. Se revisó el último fold de CORAL con pesos:
la matriz suma 559 casos y no hay predicciones de la clase 3 con una época.

**Cierre P0 parcial:** los cuatro métodos obligatorios funcionan de extremo a
extremo en `GDS_R2`.

### H4 — seis objetivos y trazabilidad

- [x] Ejecutar los cuatro métodos en `GDS_R1` a `GDS_R5`.
- [x] Ejecutar `GDS` con la excepción acordada.
- [x] Separar salidas por corrida, método, objetivo y fold.
- [x] Guardar errores de ejecución como fallos, no como resultados vacíos.
- [x] Agregar matrices de confusión por método y objetivo.
- [x] Generar tablas con media y desviación externa.
- [x] Verificar que todos los índices externos aparecen exactamente una vez.
- [x] Reproducir al menos una corrida desde un checkout limpio.

**Evidencia H4 (22/09/2026):** con `lab_pytorch` en CPU y semilla 42 se ejecutó
`python main.py --data-path 'dataset/15 atributos R0-R5.sav' --all-targets --output-dir results/h4_20260922_seed42`
(20 épocas; 5 folds externos y 3 internos, salvo `GDS` con 2 y 2). La corrida
terminó sin fallos: 24 filas método/objetivo, 108 folds externos, nueve medias
y desviaciones finitas por fila, y 27 selecciones internas de Softmax HP con
MAE y QWK finitos. `results/h4_20260922_seed42/estado.json` registra el estado;
cada método/objetivo tiene estado propio, confusión OOF agregada y artefactos
por fold. `verificacion_oof.csv` confirma 1119 índices únicos (0..1118) para
cada una de las 24 combinaciones. Los resultados de H3 no entraron en estas
tablas. La asignación índice/fold externo coincide entre los cuatro métodos
para cada objetivo.

Se repitió Softmax fijo en `GDS_R2` desde un clon local limpio y desacoplado,
en `HEAD` 56f486b, con la ruta absoluta al mismo SAV y los mismos parámetros
por defecto. Las 1119 filas OOF y la fila de nueve métricas coincidieron
exactamente. La lógica numérica de entrenamiento no cambió entre ese `HEAD` y
H4; los cambios sin commit de H4 afectan la trazabilidad y los reportes. Las
salidas generadas siguen ignoradas por Git según el plan del equipo.

**Cierre H4:** la parte experimental obligatoria funciona y tiene trazabilidad.
P0 todavía requiere resolver el alcance del protocolo, publicar las tablas
finales, completar el README y preparar el informe PDF.

### H5 — cerrar código y protocolo el 22/09

- [x] Revisar el PR [#5](https://github.com/Gael-OC/DL-2026-02-L01/pull/5):
      sigue en conflicto y sus cambios útiles de etiquetas 0..K-1 ya están en
      `main`. Su cabeza y pérdida CORAL son incompatibles con H4.
- [x] Decidir no integrar el PR #5 en su estado actual.
- [ ] Dejar una explicación breve en el PR #5 y cerrarlo **sin fusionar** como
      superado por `main`. El historial del PR queda disponible; su cierre no
      impide revisar un cambio puntual nuevo desde el código actual.
- [ ] Fijar antes de la nueva corrida el grid de cuatro configuraciones para
      CORAL sin pesos, CORN y Softmax equiparado, y las mismas cuatro con β en
      `{0.9, 0.99, 0.999}` para CORAL con pesos. Mantener 20 épocas y batch 32
      como protocolo principal del curso; no elegir el grid según el test H4.
- [ ] Permitir variar `hidden_dim` de CORAL conservando 32 como arquitectura
      base y conectar CORAL/CORAL+pesos a la búsqueda interna MAE/QWK.
- [ ] Implementar CORN: `K-1` logits, máscara condicional, pérdida con logits,
      producto acumulado y predicción; probar lotes sin ejemplos en umbrales
      superiores e integrarlo al contrato y a los folds compartidos.
- [ ] Implementar Softmax con bloque oculto equiparado a CORAL y decisión por
      mediana sobre las mismas probabilidades/checkpoint Softmax.
- [ ] Hacer un smoke test de una época en `GDS_R2` de cada método nuevo,
      verificar las salidas y ejecutar las pruebas existentes.
- [ ] Congelar métodos, grid, semillas y comando de la corrida del 23/09.

**Cierre H5:** el código de cada método que se incluirá mañana pasa pruebas y
un smoke test; no quedan cambios experimentales por decidir durante la corrida.

### H6 — corridas definitivas el 23/09

- [ ] Medir un ajuste pequeño en CPU y GPU del servidor y usar el dispositivo
      más rápido para estas MLP pequeñas.
- [ ] Ejecutar los métodos cerrados en H5 en los seis objetivos: 5×3 folds en
      `GDS_R1`–`GDS_R5` y 2×2 en `GDS`, con la agrupación 6/7 solo para
      estratificar internamente.
- [ ] Separar salidas por corrida/método/objetivo y registrar comando, commit,
      entorno, semilla, configuración seleccionada y estado de cada corrida.
- [ ] Verificar nueve métricas finitas, cuatro filas obligatorias por objetivo,
      cobertura OOF y confusiones antes de dar una corrida por definitiva.
- [ ] Comparar Softmax, CORAL y CORN dentro de cada objetivo; separar efecto de
      arquitectura, cabeza, ponderación y decisión. Si una extensión falla,
      documentarla como no incluida y conservar la entrega obligatoria.

**Cierre H6:** están disponibles y comprobados los resultados que se usarán
para README e informe; el cómputo puede seguir desatendido mientras se prepara
la documentación que no depende de las métricas finales.

### H7 — congelar resultados el 23/09

- [ ] Seleccionar la corrida definitiva de cada método y objetivo y registrar
      commit, entorno, semilla, comandos y folds usados.
- [ ] Verificar que tablas, figuras y comandos corresponden al mismo código.
- [ ] Copiar solo las tablas y figuras finales a una ubicación versionada;
      `results/` conserva los artefactos locales de corrida y sigue ignorado.
- [ ] Comparar métodos dentro de cada objetivo; no usar el ranking global para
      proclamar un ganador entre objetivos de distinta dificultad.
- [ ] Registrar limitaciones, resultados favorables y desfavorables, incluidos
      desbalance, patrones ambiguos y la excepción de `GDS`.
- [ ] No abrir nuevas búsquedas motivadas por los resultados del test externo.

**Cierre:** no cambia el código experimental salvo correcciones justificadas.

### H8 — completar README el 23/09

- [x] Dejar un esqueleto del `README.md` siguiendo la organización de los
      laboratorios anteriores, sin inventar resultados finales.
- [ ] Documentar instalación, dataset, seis objetivos y estructura real.
- [ ] Documentar métodos, nested CV, grid efectivo, seeds y excepción de
      `GDS`; CORN aparece solo si H6 se completó.
- [ ] Incorporar comandos, tablas y figuras definitivas desde H7.
- [ ] Interpretar métricas, errores por clase y limitaciones dentro de cada
      objetivo, y redactar las conclusiones.
- [ ] Incorporar el ejemplo numérico breve de CORAL y revisar que la
      explicación de CORAL y pesos coincida con el código.
- [ ] Comprobar que el README permite reconstruir el procedimiento completo.

**Cierre H8:** el README es la fuente consolidada para redactar el informe.

### H9 — informe en Overleaf y entrega el 24/09

- [ ] Redactar el informe en Overleaf a partir del README final y los
      artefactos congelados. Esta tarea no incluye generar el informe ahora.
- [ ] Revisar resultados, método y explicación de CORAL en el borrador; una
      segunda lectura es bienvenida si está disponible.
- [ ] Verificar coherencia entre código, README, PDF y ZIP.
- [ ] Confirmar acceso al repositorio para profesor/ayudantes.
- [ ] Registrar la entrega en Campus Virtual.

**Cierre final:** otra persona puede ir del comando a la tabla sin información
oral adicional.

## 7. Flujo de GitHub

Ramas sugeridas:

```text
gael/core-validacion
gael/nested-cv
gael/corn
johan/coral
johan/coral-weights
johan/reporting
integracion/p0
```

Cada pull request debe indicar:

```text
Objetivo:
Archivos modificados:
Cómo se comprobó:
Comando ejecutado:
Resultado o artefacto:
Pendientes conocidos:
```

Reglas de integración:

1. Actualizar la rama desde `main` antes de solicitar integración.
2. No mezclar una funcionalidad con refactors o formato no relacionados.
3. No subir el archivo SAV ni artefactos masivos de ejecución.
4. No sobrescribir salidas de otra corrida.
5. Marcar el checklist después de integrar el cambio, no solo después de
   escribirlo localmente.
6. Los cambios que afecten el motor o el protocolo se revisan en un punto de
   integración compartido antes de incorporarlos a `main`.

## 8. Registro breve de decisiones

Añadir una fila solamente para decisiones que cambien el procedimiento.

| Fecha | Decisión | Motivo | Participantes | Evidencia |
|---|---|---|---|---|
| 16/09/2026 | CORN será la extensión principal | Permite una comparación ordinal coherente | Ambos; implementación de Gael | Plan de ejecución |
| 16/09/2026 | Agrupar clases 6 y 7 solo para estratificar la validación interna de `GDS` | Solo existen dos observaciones en la clase extrema | Gael | Protocolo acordado |

## 9. Estado diario

Actualizar de forma breve; los detalles pertenecen al pull request o al
artefacto correspondiente.

| Fecha | Persona | Terminado | Siguiente | Bloqueo |
|---|---|---|---|---|
| 16/09/2026 | Gael | H0 local | Validación y folds | — |
| 16/09/2026 | Johan | — | CORAL y pruebas sintéticas | — |

## 10. Retiro de este documento

Este archivo puede eliminarse de la versión final cuando se cumplan todas las
condiciones siguientes:

- [ ] P0 y las extensiones que efectivamente se reportarán están cerradas.
- [ ] Las decisiones relevantes fueron traspasadas al `README.md`.
- [ ] El README contiene los comandos, versiones, seeds y protocolo final.
- [ ] El README contiene tablas, figuras, interpretación y limitaciones.
- [ ] Los PR y commits conservan la trazabilidad del desarrollo.
- [ ] El informe fue construido a partir de la misma versión documentada.

Eliminar el archivo del árbol final no borra su historial de Git. No debe
eliminarse antes de completar el traspaso al README.
