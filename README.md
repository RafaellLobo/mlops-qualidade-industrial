# Estabilidade de rede elétrica — CI/CD/CT para ML

Adaptação individual de Rafael Lobo do projeto didático de
[Wesley Santos](https://github.com/wesleyasantos/ci_cd_ct_mlops), sem fork e com histórico próprio.
Objetivo: detectar **instabilidade** em uma rede elétrica simulada de quatro nós e apoiar
a triagem de cenários por pesquisadores/engenheiros. Demonstração acadêmica; sem controle de rede real.

## Dados e protocolo

[UCI Electrical Grid Stability Simulated Data](https://doi.org/10.24432/C5PG66),
Vadim Arzamasov, 2018, **CC BY 4.0**, 10.000 simulações, 12 atributos numéricos, sem dados pessoais.
Fonte original, atribuição e hash estão em `data/source/`.

Entradas na ordem: `tau1,tau2,tau3,tau4,p1,p2,p3,p4,g1,g2,g3,g4`.
`tau`: tempo de reação; `p`: potência produzida/consumida; `g`: elasticidade.
`p1` é positivo, consumidores `p2..p4` negativos e a soma deve ser zero.
`stab` determina o rótulo e é **excluído**; `stabf` é convertido no alvo `instavel`: stable=0, unstable=1.

Divisão estratificada fixa (semente 7): **6.000 treino / 2.000 holdout / 2.000 reserva**.
Sem sobreposição; reserva gera lotes, cada lote é dividido em metades treino/avaliação pelo DVC.
Nenhuma busca de hiperparâmetros utiliza o holdout. O holdout é referência didática reutilizada
pelo portão; produção exigiria avaliação final independente e renovação controlada da referência.
Os lotes começaram vazios na adaptação obrigatória. O desafio adicional incorpora na main
o lote aprovado em E3: 200 registros entram no treino e 200 na avaliação, com divisão fixa.

## Contrato e modelo

Classe positiva **1 = instável**. Recall **>=0,92**, precisão **>=0,90** e recall candidato
**>= campeão no mesmo conjunto**, sem tolerância a regressão. Piso de sanidade do campeão: 0,80.
Falso negativo é mais caro: uma instabilidade ignorada deixa de ser investigada.
Precisão limita investigações desnecessárias. São metas acadêmicas propostas antes da avaliação,
sem equivalência a limites de segurança elétrica ou custos financeiros comprovados.

HistGradientBoostingClassifier (500 iterações, semente 42) em pipeline scikit-learn,
com hiperparâmetros em `params.yaml`. Árvores dispensam StandardScaler; restrições monotônicas
em g1..g4 explicitam o comportamento esperado e são testadas. Não se promete causalidade.
Versões diretas fixadas no `pyproject.toml`; `requirements-lock.txt` registra a resolução
do ambiente, com marcador de plataforma para pywin32. Usado em CI, CT e Docker.

## Executar localmente (PowerShell, Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -c requirements-lock.txt -e ".[dev,ct]"
python demo/local.py ci
python demo/local.py ct
uvicorn mlops_demo.api:app
# outro terminal, ambiente ativo:
python demo/smoke_test.py http://localhost:8000
```

Os CSVs e campeão já são versionados. `python -m mlops_demo.data base` só cria bases ausentes
e recusa sobrescrever. Para reproduzir a divisão, use `dividir_dataset()` e compare os derivados;
a fonte local dispensa rede. Para recriar intencionalmente o campeão: `python -m mlops_demo.train --campeao`.
`dvc.lock` é versionado; o cache DVC é local e os dados pequenos ficam no Git.

## Pipelines no GitHub

| Pipeline | Gatilho | Verificação |
|---|---|---|
| CI | main e PRs | ruff; dados; código, dados, modelo e API, Python 3.11 e 3.12 |
| CT | lotes/params em PR/main; semanal; manual | KS por atributo; DVC; comparação e portão; promoção por PR |
| CD | tag v*.*.* | reutiliza CI; build; smoke no container; publicação GHCR |

Actions configurado com escrita e permissão de abrir PRs; variável de repositório
`PROMOCAO_AUTOMATICA=true`. Nenhum token é versionado: somente `GITHUB_TOKEN` do Actions.
O desafio adicional ativou a promoção automática na main sem proteção. Em PR, CT avalia sem promover.
Na main, o candidato aprovado é promovido diretamente; o CT cria a tag e dispara o CD.

Drift: KS p<0,01 em **pelo menos 2 dos 12 atributos**. O cenário 4 seleciona 400 registros
do quartil superior de elasticidade média da reserva, sem modificar medições ou rótulos.
O gatilho detecta mudança estatística; isso não demonstra piora nem ganho de desempenho.

## Cenários e evidências

Crie branch a partir da main; aplique `python demo/cenarios.py aplicar N` e abra PR.
Falhas controladas permanecem em branches e não devem ser integradas.

| Cenário | Mudança | Resultado esperado |
|---|---|---|
| 1 | remove g4 da matriz de entrada | CI vermelho em código |
| 2 | tau1 nulo e g1 negativo | validação de dados falha |
| 3 | max_iter=1 | modelo abaixo do contrato |
| 4 | seleção de alta elasticidade da reserva | drift; retreino; portão com tabela |
| 5 | peso 1.000.000 para instável | recall alto e precisão baixa; gate bloqueia |

Para E4, aplique cenários 4 e 5 na mesma branch: o lote dispara o CT por evento.
Ver [ADAPTACAO.md](ADAPTACAO.md) para links E1–E5 e limites.

### Resultado da adaptação

As cinco evidências foram executadas. O primeiro CT com drift e 150 iterações foi bloqueado
por perder recall para o campeão ([PR #3](https://github.com/RafaellLobo/mlops-qualidade-industrial/pull/3)).
Após autorização, compararam-se 150/300/500 iterações somente numa validação interna
de 4.800/1.200 linhas de treino (seed 42); foram selecionadas 500.
`python demo/selecionar_params.py` reproduz a seleção sem ler holdout ou lotes e grava
`docs/selecao_validacao.json`. Para reproduzir E3, use os parâmetros e o lote do
[PR #5](https://github.com/RafaellLobo/mlops-qualidade-industrial/pull/5): recall 0,9426 e precisão 0,9612,
acima do recall 0,9356 do campeão no mesmo conjunto de 2.200 registros.
O cenário 4 sozinho detecta drift, mas aprovação do portão depende do resultado medido.

O campeão inicial de 150 iterações permanece preservado na tag e imagem 1.0.0.
No desafio adicional, a main recebeu o lote de E3 e parâmetros de 500 iterações;
o CT promoveu automaticamente o candidato e publicou a versão 1.1.0.
As branches e execuções das evidências obrigatórias foram preservadas.
O script de seleção não altera `params.yaml` automaticamente. Não há busca de parâmetros
no CT: seus hiperparâmetros vêm do arquivo versionado no respectivo PR.

| Evidência | Execução |
|---|---|
| E1 | [Actions](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45409481397) |
| E2 | [Actions](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45409558492) |
| E3 | [Actions](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45410898898) |
| E4 | [Actions](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45410188783) |
| E5 | [Actions](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45409892618) |

## API e container

`GET /health`: estado, versão e métricas. `POST /predict`: lista `atributos` com exatamente
12 números finitos na ordem documentada; faixas e balanço físico também são validados.
Resposta: `classe` (estavel/instavel), `probabilidade_instavel` e `versao_codigo`.
Entradas inválidas retornam HTTP 422.

```powershell
docker run --rm -p 8000:8000 ghcr.io/rafaelllobo/mlops-qualidade-industrial:1.1.0
python demo/smoke_test.py http://localhost:8000
```

A visibilidade do pacote GHCR foi verificada como **Public** em 24/11/2026; o pull anônimo
e o smoke test local passaram. [Página do pacote](https://github.com/users/RafaellLobo/packages/container/package/mlops-qualidade-industrial).
O repositório público por si só não garante a visibilidade pública do pacote.


## Desafio adicional: promoção automática concluída

Configuração verificada: `PROMOCAO_AUTOMATICA=true`, main sem proteção/rulesets ativos,
Actions com escrita de conteúdo e permissão de disparar workflows. A configuração já
prevista no projeto foi usada; nenhum workflow precisou ser alterado.

1. [PR #7](https://github.com/RafaellLobo/mlops-qualidade-industrial/pull/7): reutilizou exatamente lote e parâmetros aprovados em E3.
2. [CT automático da main](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45414911766): detectou drift em g1..g4, reproduziu o DVC e aprovou o portão.
3. [Commit do robô](https://github.com/RafaellLobo/mlops-qualidade-industrial/commit/a5f296f25e31ee8a989994975d1b400ebd347ea8): alterou somente `models/champion/`.
4. [Tag v1.1.0](https://github.com/RafaellLobo/mlops-qualidade-industrial/tree/v1.1.0): criada pelo CT sobre esse mesmo commit.
5. [CD disparado pelo CT](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45415087946): evento `workflow_dispatch`, ator `github-actions[bot]`,
   CI, construção, smoke e publicação aprovados. O disparo explícito é necessário porque
   a tag criada com `GITHUB_TOKEN` não inicia outro workflow por evento push.

Candidato: recall 0.9426,
precisão 0.9612, 2.200 amostras de avaliação;
campeão anterior: recall 0.9356, mesmo conjunto.
O treino usa 6.200 registros, sem mudar holdout, sementes ou limiares.
Pull anônimo e smoke local da imagem 1.1.0 aprovados; digest em `docs/desafio_automatico.json`.

**Vantagens:** reduz intervenção manual, mantém trilha de commit/tag/imagem e entrega
rapidamente um modelo que cumpre o contrato. **Riscos:** main sem proteção e ausência
de revisão humana permitem propagar falhas não cobertas pelo portão; métricas agregadas
e holdout reutilizado não garantem segurança operacional. Em produção, usaríamos aprovação
por ambiente, validação independente e processo de rollback. Aqui `true` permanece ativo
para a demonstração; para retornar à revisão humana, definir a variável como `false`.

O lote já está na main: não tente recriar `lote_01.csv`. Use o PR #7 para reprodução
histórica. No cenário 4, um novo lote exige nome único e reserva não sobreposta.

## Atribuição e limites

Código de partida: repositório de Wesley Santos, preservado no commit inicial deste histórico;
o original não declara licença no material clonado. A licença CC BY 4.0 se aplica aos dados UCI.
Rafael Lobo é o responsável pela entrega individual. Codex foi usado para inspeção dos labs,
adaptação, testes, documentação e execução dos pipelines. Os resultados são evidências técnicas
da demonstração, sem validação operacional. O enunciado pede atividade em grupo; a aceitação
da modalidade individual deve ser confirmada com o professor, sem atribuir commits a terceiros.
