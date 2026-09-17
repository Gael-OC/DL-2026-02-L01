# Plan de ejecución — Laboratorio 01

> Documento temporal de coordinación del equipo. Se mantiene durante el
> desarrollo y se elimina de la versión final solamente después de traspasar
> al `README.md` el procedimiento, los comandos, las decisiones y los
> resultados definitivos.

**Equipo:** Gael y Johan  
**Entrega:** 24/09/2026 a las 23:59

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

Para elevar el laboratorio se implementará **CORN** como extensión principal.
Después de completar lo obligatorio se evaluarán Softmax con arquitectura
equiparada y la decisión por mediana. No se añadirán más modelos mientras la
entrega obligatoria no esté cerrada.

## 2. Forma de trabajo

- Gael se concentra en validación experimental, flujo común y CORN.
- Johan se concentra en CORAL, ponderación, salidas y reportes.
- La asignación de componentes evita ediciones simultáneas.
- Los cambios llegan a `main` mediante ramas cortas y pull requests.
- Un cambio se incorpora cuando tiene una comprobación reproducible y ambos
  conocen su efecto sobre el experimento.
- El test externo nunca se usa para escoger configuración, época, método o
  decodificador.
- Los seis objetivos son experimentos independientes; no son una red con seis
  salidas.
- El informe se redactará después de congelar los resultados. Durante el
  desarrollo solo se guardará evidencia breve y verificable.

## 3. Distribución de tareas

### Gael — validación experimental, flujo común y CORN

Archivos asignados para evitar solapamientos:

- `main.py`
- `src/config.py`
- `src/preprocessing.py`
- `src/evaluation.py`
- implementación de CORN

Tareas:

- validar el contrato de datos y las seis recodificaciones;
- generar y conservar folds reproducibles;
- asegurar que no exista fuga del test externo;
- corregir métricas para conservar la escala completa de clases;
- implementar la búsqueda interna por menor MAE y desempate por mayor QWK;
- construir el motor común de entrenamiento y evaluación;
- definir el contrato que deben cumplir Softmax, CORAL y CORN;
- conectar al flujo común los métodos que cumplan el contrato compartido;
- implementar y probar CORN;
- congelar el protocolo antes de las ejecuciones finales.

### Johan — CORAL, ponderación, salidas y reportes

Archivos asignados para evitar solapamientos:

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
- ejecutar comandos acordados y registrar sus artefactos;
- implementar Softmax equiparado y decisión por mediana si P0 ya está cerrado.

### Regla para cambios fuera del área propia

Si una tarea necesita modificar un archivo asignado a la otra persona, se
acuerda primero la interfaz. No se hacen cambios paralelos sobre el mismo
archivo. Quien implementa un método no necesita modificar `main.py`: entrega
el modelo, la pérdida y el decodificador a través del contrato común. El
resultado se comprueba en conjunto al integrar el método en el experimento.

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

- [ ] Implementar `labels_to_levels` para `K=2`, `K=3` y `K=7`.
- [ ] Implementar `CoralLayer` con umbrales ordenados.
- [ ] Implementar `MLPCoral`.
- [ ] Implementar la pérdida CORAL sin aplicar sigmoid previamente.
- [ ] Implementar `logits_to_ordinal_predictions`.
- [ ] Añadir pruebas de formas, finitud y gradientes.

**Cierre:** la base Softmax conserva folds correctos y CORAL supera pruebas
sintéticas sin depender todavía de `main.py`.

### H2 — búsqueda interna y ponderación

#### Gael

- [ ] Aplicar en `GDS` la agrupación de las clases originales 6 y 7 solo
      para estratificar los folds internos, conservando las etiquetas
      originales para entrenamiento y evaluación.
- [ ] Recorrer las cuatro configuraciones de `HYPERPARAMETER_GRID`.
- [ ] Promediar MAE y QWK sobre todos los folds internos.
- [ ] Elegir menor MAE; en empate, mayor QWK.
- [ ] Reentrenar desde cero sobre todo outer-train.
- [ ] Guardar la mejor configuración de cada outer fold.
- [ ] Verificar con un caso controlado que outer-test no influye en la selección.

#### Johan

- [ ] Implementar pesos por número efectivo y normalizarlos a media uno.
- [ ] Manejar clases ausentes sin divisiones por cero ni `NaN`.
- [ ] Aplicar el peso de la clase de cada muestra a la pérdida CORAL.
- [ ] Comprobar que pesos iguales a uno reproducen la pérdida no ponderada.
- [ ] Preparar un ejemplo numérico breve de CORAL para documentación posterior.

**Cierre:** Softmax HP funciona correctamente y CORAL ponderado pasa sus
pruebas unitarias.

### H3 — integración P0 en `GDS_R2`

- [ ] Integrar Softmax fijo mediante el contrato común.
- [ ] Integrar Softmax HP mediante el contrato común.
- [ ] Integrar CORAL sin pesos.
- [ ] Integrar CORAL con pesos.
- [ ] Producir cuatro filas comparables para `GDS_R2`.
- [ ] Guardar la configuración elegida por outer fold.
- [ ] Guardar predicciones OOF y probabilidades.
- [ ] Revisar manualmente una matriz de confusión y una corrida.

**Cierre P0 parcial:** los cuatro métodos obligatorios funcionan de extremo a
extremo en `GDS_R2`.

### H4 — seis objetivos y trazabilidad

- [ ] Ejecutar los cuatro métodos en `GDS_R1` a `GDS_R5`.
- [ ] Ejecutar `GDS` con la excepción acordada.
- [ ] Separar salidas por corrida, método, objetivo y fold.
- [ ] Guardar errores de ejecución como fallos, no como resultados vacíos.
- [ ] Agregar matrices de confusión por método y objetivo.
- [ ] Generar tablas con media y desviación externa.
- [ ] Verificar que todos los índices externos aparecen exactamente una vez.
- [ ] Reproducir al menos una corrida desde un checkout limpio.

**Cierre P0:** existe una entrega completa aunque no se alcance ninguna
extensión.

### H5 — CORN y comparación controlada

#### Gael

- [ ] Implementar `MLPCorn` con `K-1` logits.
- [ ] Implementar máscara de subtareas condicionales.
- [ ] Implementar pérdida CORN con logits.
- [ ] Implementar producto acumulado para probabilidades.
- [ ] Probar lotes con subtareas superiores vacías.
- [ ] Integrar CORN al mismo grid y a los mismos folds.

#### Johan

- [ ] Implementar Softmax con el bloque oculto equiparado.
- [ ] Implementar decisión por mediana desde probabilidades Softmax.
- [ ] Comprobar argmax y mediana sobre el mismo checkpoint.
- [ ] Extender tablas y gráficos sin mezclar objetivos de distinta dificultad.

- [ ] Ejecutar CORN en los seis objetivos.
- [ ] Comparar Softmax, CORAL y CORN dentro de cada objetivo.
- [ ] Separar efecto de arquitectura, cabeza, pesos y regla de decisión.

**Cierre P1:** las extensiones responden una pregunta experimental clara y no
son solo modelos agregados.

### H6 — congelar resultados

- [ ] Seleccionar las corridas definitivas antes de redactar conclusiones.
- [ ] Verificar que tablas, figuras y comandos corresponden al mismo código.
- [ ] Copiar las tablas y figuras finales a una ubicación versionada.
- [ ] Registrar limitaciones, resultados favorables y resultados desfavorables.
- [ ] No abrir nuevas búsquedas después de observar el test externo.

**Cierre:** no cambia el código experimental salvo correcciones justificadas.

### H7 — README, informe y entrega

- [ ] Documentar instalación y entorno en `README.md`.
- [ ] Documentar dataset, entradas, objetivos y recodificaciones.
- [ ] Documentar nested CV, grid, seeds y excepción de `GDS`.
- [ ] Documentar Softmax, CORAL, ponderación y CORN.
- [ ] Incluir comandos exactos de reproducción.
- [ ] Incluir resultados, figuras, discusión y limitaciones.
- [ ] Verificar que el README permite reconstruir el procedimiento completo.
- [ ] Redactar el informe usando el README y los artefactos congelados.
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
