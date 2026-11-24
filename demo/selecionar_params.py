"""Seleção curta autorizada, usa apenas base_treino; nunca abre holdout/lotes."""

import copy
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_demo import config
from mlops_demo.evaluate import avaliar
from mlops_demo.train import treinar

BASE_PARAMS = {
    "semente": 42,
    "modelo": {
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 5,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
    },
}
df = pd.read_csv(config.BASE_TREINO)
train, valid = train_test_split(df, test_size=0.2, stratify=df[config.ALVO], random_state=42)
resultados = []
for iters in (150, 300, 500):
    params = copy.deepcopy(BASE_PARAMS)
    params["modelo"]["max_iter"] = iters
    metrics = avaliar(treinar(train, params), valid)
    resultados.append({"params": params, "validacao_interna": metrics})
    print(iters, metrics, flush=True)
elegiveis = [r for r in resultados if r["validacao_interna"]["precisao_instavel"] >= 0.90]
if not elegiveis:
    raise SystemExit("Nenhuma configuração atingiu precisão mínima na validação interna.")
best = max(
    elegiveis,
    key=lambda r: (
        r["validacao_interna"]["recall_instavel"],
        r["validacao_interna"]["precisao_instavel"],
        -r["params"]["modelo"]["max_iter"],
    ),
)
out = {
    "protocolo": "split estratificado 4800/1200 apenas de base_treino, random_state=42",
    "criterio": "maior recall entre precisao>=0.90; desempate por precisao, depois menor max_iter",
    "holdout_usado_na_selecao": False,
    "resultados": resultados,
    "selecionado": best["params"],
}
(config.RAIZ / "docs" / "selecao_validacao.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8"
)
print("SELECIONADO:", best["params"], flush=True)
