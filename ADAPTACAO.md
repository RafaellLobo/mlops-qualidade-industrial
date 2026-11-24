# Adaptação — estabilidade de rede elétrica

**Responsável: Rafael Lobo · Entrega: 25/11/2026 · Modalidade individual solicitada.**

**1. Problema e dados.** Classificar cenários de uma rede simulada de quatro nós como estáveis
ou instáveis, para triagem por pesquisadores de sistemas elétricos. Fonte: Vadim Arzamasov (2018),
[UCI Electrical Grid Stability Simulated Data](https://doi.org/10.24432/C5PG66), CC BY 4.0.
São 10.000 linhas e 12 atributos numéricos (tau1..4, p1..4, g1..4), sem dados pessoais.
`stab` é excluído por determinar o rótulo; `stabf` vira `instavel` (stable=0, unstable=1).
Divisão estratificada, semente 7: 6.000 treino, 2.000 holdout, 2.000 reserva, disjuntos.
Fonte e SHA-256 preservados em `data/source/`. Labs baixados 05/06 usam Iris/Wine; 07 usa Breast Cancer.

**2. Contrato.** Recall da classe **1 (instável) >=0,92**, precisão **>=0,90** e recall do candidato
não inferior ao campeão no mesmo conjunto. Falso negativo é mais caro por deixar uma instabilidade
sem investigação; precisão evita alertas excessivos. Metas didáticas propostas antes da avaliação,
sem custo financeiro ou requisito de segurança real validado. Piso de sanidade do campeão: 0,80.

**3. Alterações.** Mantidos pacote `mlops_demo`, arquitetura e três workflows do projeto
[de partida](https://github.com/wesleyasantos/ci_cd_ct_mlops); histórico reiniciado, sem fork.

| Arquivos | Adaptação |
|---|---|
| `data/raw/base_*.csv`, `data/source/*` | UCI, partições, fonte e proveniência; lotes iniciais vazios |
| `data.py` | fonte local, geração fixa, seleção por elasticidade, esquema/faixas/sinais/balanço |
| `config.py` | 12 atributos, alvo/classe positiva, g1..4, contrato e KS em 2 atributos |
| `features.py`, `params.yaml`, `train.py` | HGB, hiperparâmetros externos; treino existente reutilizado |
| `evaluate.py`, `gate.py` | recall/precisão de instável e tabela com nomes do domínio |
| `monitor.py`, `demo/gerar_lote.py` | KS reutilizado; seleção de 400 casos de alta elasticidade |
| `api.py`, `demo/smoke_test.py` | 12 entradas, validação física e resposta estavel/instavel |
| `tests/test_{codigo,dados,modelo,api}.py` | camadas adaptadas e novos testes de seleção, balanço, vazamento e HTTP 422 |
| `demo/cenarios.py` | cinco cenários próprios; falha de peso extremo usa classe 1 |
| `models/champion/*` | primeiro campeão treinado somente na base de treino |
| `dvc.yaml`, `dvc.lock` | invalidação por config e pipeline reproduzível |
| `pyproject.toml`, `requirements-lock.txt`, workflows, `Dockerfile` | versões fixadas e compartilhadas |
| `README.md`, `ADAPTACAO.md` | domínio, reprodução, evidências, autoria e limites |

**4. Evidências verificáveis.** Baseline original, antes de adaptar:
[CI com três jobs verdes](https://github.com/RafaellLobo/mlops-qualidade-industrial/actions/runs/45406647880).

| Evidência | Execução |
|---|---|
| E1 — CI main, três jobs verdes | A preencher após execução |
| E2 — CI vermelho em PR, remoção de g4 | A preencher após execução |
| E3 — CT com drift e portão aprovado no PR | A preencher após execução |
| E4 — CT com precisão insuficiente, sem promoção | A preencher após execução |
| E5 — CD da tag v1.0.0 e publicação GHCR | A preencher após execução |

**5. Limites e riscos.** O portão cobre duas métricas agregadas, não calibração, subgrupos,
segurança ou causalidade. KS não prova degradação; múltiplos testes podem gerar alarmes.
Holdout reutilizado pelo gate não substitui teste final independente. Em produção monitoraria
recall/precisão com rótulos atrasados, calibração, drift, balanço físico, prevalência, latência,
erros e qualidade por faixa operacional. Reserva finita não deve ser reamostrada em lotes
sobrepostos; promoção repetida exige rastrear amostras já vistas pelo campeão.
Promoção automática permanece desligada e falhas controladas não são integradas à main.

**6. Contribuições e IA.** Rafael Lobo: responsável pelo repositório, execução e entrega.
Codex auxiliou a inspeção dos labs, escolha e preparação da fonte, implementação, testes,
documentação e evidências Actions. Não há atribuição de trabalho a integrantes inexistentes.
O enunciado prevê grupo; confirmar com o professor a aceitação da entrega individual.
