"""Testes de dados: esquema, valores e vazamento entre treino e teste."""

import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import listar_lotes, validar_dataframe

pytestmark = pytest.mark.dados


def _arquivos_de_dados():
    return [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]


@pytest.mark.parametrize("caminho", _arquivos_de_dados(), ids=lambda c: c.name)
def test_arquivo_respeita_esquema_e_valores(caminho):
    problemas = validar_dataframe(pd.read_csv(caminho))
    assert problemas == [], f"{caminho.name}: {problemas}"


def test_base_de_treino_tem_as_duas_classes_em_proporcao_razoavel(base_treino):
    proporcao_instavel = (base_treino[config.ALVO] == config.CLASSE_INSTAVEL).mean()
    assert 0.60 <= proporcao_instavel <= 0.67


def test_nao_ha_vazamento_entre_treino_e_holdout(base_treino, base_holdout):
    chaves_treino = set(map(tuple, base_treino[config.FEATURES].round(6).to_numpy()))
    chaves_holdout = set(map(tuple, base_holdout[config.FEATURES].round(6).to_numpy()))
    assert chaves_treino.isdisjoint(chaves_holdout)


def test_nao_ha_linhas_duplicadas_na_base_de_treino(base_treino):
    assert not base_treino.duplicated().any()


def test_validador_detecta_nulos_e_valores_negativos(base_treino):
    corrompido = base_treino.copy()
    corrompido.loc[0, "tau1"] = None
    corrompido.loc[1, "g1"] = -5.0
    problemas = validar_dataframe(corrompido)
    assert any("nulos" in p for p in problemas)
    assert any("fora da faixa" in p for p in problemas)


def test_validador_detecta_esquema_diferente(base_treino):
    assert validar_dataframe(base_treino.drop(columns=["tau1"]))


def test_balanco_de_potencia_e_sinais_fisicos(base_treino):
    assert validar_dataframe(base_treino) == []  # p2..p4 negativos são válidos
    invalido = base_treino.copy()
    invalido.loc[0, "p1"] += 0.1
    assert any("balanço" in p for p in validar_dataframe(invalido))


def test_validador_rejeita_texto_e_infinito(base_treino):
    texto = base_treino.astype({"tau1": "object"})
    texto.loc[0, "tau1"] = "inválido"
    assert any("não numéricas" in p for p in validar_dataframe(texto))
    infinito = base_treino.copy()
    infinito.loc[0, "tau1"] = float("inf")
    assert any("não finitos" in p for p in validar_dataframe(infinito))


def test_reserva_e_bases_sao_disjuntas():
    from mlops_demo.data import dividir_dataset

    partes = dividir_dataset()
    chaves = [set(map(tuple, d[config.FEATURES].round(6).to_numpy())) for d in partes]
    assert [len(d) for d in partes] == [6000, 2000, 2000]
    assert all(chaves[i].isdisjoint(chaves[j]) for i in range(3) for j in range(i + 1, 3))
