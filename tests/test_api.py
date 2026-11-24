"""Testes de contrato da API: formato de entrada e de saída do /predict."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_devolve_o_formato_combinado(base_holdout):
    estado = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": estado})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_instavel", "versao_codigo"}
    assert corpo["classe"] in {"instavel", "estavel"}
    assert 0.0 <= corpo["probabilidade_instavel"] <= 1.0


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


def test_predict_acerta_um_caso_instavel_e_um_estavel_claros(base_holdout, campeao):
    """Escolhe os casos mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_INSTAVEL)
    p_instavel = campeao.predict_proba(X)[:, coluna]
    i_instavel = int(p_instavel.argmax())
    i_estavel = int(p_instavel.argmin())
    for indice, esperado in ((i_instavel, "instavel"), (i_estavel, "estavel")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado


def test_api_rejeita_potencia_sem_balanco(base_holdout):
    estado = base_holdout[config.FEATURES].iloc[0].tolist()
    estado[4] += 0.1
    assert cliente.post("/predict", json={"atributos": estado}).status_code == 422


def test_api_rejeita_elasticidade_impossivel(base_holdout):
    estado = base_holdout[config.FEATURES].iloc[0].tolist()
    estado[8] = -0.1
    assert cliente.post("/predict", json={"atributos": estado}).status_code == 422
