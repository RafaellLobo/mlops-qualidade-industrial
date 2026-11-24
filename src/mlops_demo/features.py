"""Atributos físicos e classificador de estabilidade da rede elétrica."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from mlops_demo.config import ALVO, FEATURES


def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    faltando = [c for c in FEATURES + [ALVO] if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")
    X = df[FEATURES[:-1]].copy()
    return X, df[ALVO].astype(int)


def construir_pipeline(params: dict) -> Pipeline:
    # Árvores dispensam padronização. Elasticidade maior não deve reduzir o risco.
    return Pipeline(
        [
            (
                "classificador",
                HistGradientBoostingClassifier(
                    **params["modelo"],
                    random_state=params["semente"],
                    early_stopping=False,
                    monotonic_cst=[0] * 8 + [1] * 4,
                ),
            )
        ]
    )
