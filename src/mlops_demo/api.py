"""API de inferência (FastAPI) servindo o modelo campeão.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlops_demo import __version__, config
from mlops_demo.data import validar_dataframe

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

app = FastAPI(title="Estabilidade de rede elétrica (CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class EstadoRede(BaseModel):
    """12 atributos: tau1..4, p1..4, g1..4, nesta ordem."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    atributos: list[float] = Field(
        min_length=12, max_length=12, description=", ".join(config.FEATURES)
    )

    @model_validator(mode="after")
    def validar_estado(self):
        df = pd.DataFrame([self.atributos], columns=config.FEATURES)
        df[config.ALVO] = 0
        problemas = validar_dataframe(df)
        if problemas:
            raise ValueError("; ".join(problemas))
        return self


class Predicao(BaseModel):
    classe: str
    probabilidade_instavel: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(estado: EstadoRede) -> Predicao:
    entrada = pd.DataFrame([estado.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    # a coluna da classe instavel (1) em predict_proba segue a ordem de classes_
    coluna = list(_modelo.classes_).index(config.CLASSE_INSTAVEL)
    p_instavel = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="instavel" if previsto == config.CLASSE_INSTAVEL else "estavel",
        probabilidade_instavel=round(p_instavel, 4),
        versao_codigo=__version__,
    )
