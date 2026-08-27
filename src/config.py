"""Constantes simples para mantener el proyecto ordenado."""

FEATURE_COLUMNS = [
    "Día",
    "Mes",
    "Año",
    "Estación",
    "País",
    "Ciudad",
    "CalleLugar",
    "NumeroPiso",
    "Miguel2",
    "González2",
    "Avenida2",
    "Imperial2",
    "A682",
    "Caldera2",
    "Copiapo2",
]

TARGET_COLUMNS = [
    "GDS",
    "GDS_R1",
    "GDS_R2",
    "GDS_R3",
    "GDS_R4",
    "GDS_R5",
]

ID_COLUMN = "ID"

DEFAULT_TARGET = "GDS_R2"
DEFAULT_HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.15
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_RANDOM_SEED = 42

# TODO(alumno): recorrer este grid en el loop interno de validacion.
# Seleccionar la configuracion con menor MAE interno (empate: mayor QWK).
# No usar el fold externo para elegir hiperparametros.
# Para CORAL, incluir tambien "beta" en cada diccionario.
HYPERPARAMETER_GRID = [
    {
        "hidden_dim": 32,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
    {
        "hidden_dim": 64,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
    {
        "hidden_dim": 32,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
    {
        "hidden_dim": 32,
        "dropout": 0.15,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
    },
]
