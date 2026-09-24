# Deploy de Modelo de ML : 2 Abordagens

Um modelo de Machine Learning, implantado de 2 formas diferentes : batch
agendado e API containerizada  para comparar arquiteturas de deploy na
prática.

🔗 **[Testar o Deploy 2 (API + Interface) ao vivo](https://deployml-onpremise.onrender.com)**
*(plano gratuito do Render - primeira requisição pode levar até 1 minuto, cold start)*

![Interface do Deploy 2 em uso](docs/images/deploy2-interface.png)

---

## 1. Problema

Um pipeline Scikit-learn (`ColumnTransformer` + `TargetEncoder` +
`LGBMClassifier`) prevê se um cliente tem perfil de compra **Online** ou em
**Loja física**, a partir de 24 variáveis comportamentais e demográficas.

Ter um modelo treinado não é o mesmo que ter um modelo **em produção**. O
mesmo `.pkl` foi implantado de 2 formas, cada uma resolvendo um cenário de
negócio distinto:

| Deploy | Cenário de uso | Status |
|---|---|---|
| **1. Batch (Databricks)** | Previsão diária em massa, sem necessidade de resposta imediata | ✅ Completo |
| **2. API + Interface (Docker → Render)** | Uso interativo, um cliente por vez, resposta em tempo real | ✅ Completo |

---

## 2. Arquitetura

![Diagrama: um modelo, duas arquiteturas de deploy](docs/images/diagrama-arquiteturas.svg)

**Por que 2 formas de deploy do mesmo modelo?** Não existe "a melhor forma
de fazer deploy de ML" — existe a forma certa pra cada contexto:

- **Batch** processa grande volume de uma vez, sem urgência de resposta :
  ideal quando a decisão pode esperar até o próximo ciclo agendado.
- **API + Interface** responde a um cliente por vez, com baixa latência :
  ideal quando alguém (humano ou sistema) precisa da previsão na hora.

| Critério | Batch (Databricks) | API (Docker/Render) |
|---|---|---|
| Latência | Alta (agendado, não sob demanda) | Baixa (segundos) |
| Custo em ociosidade | Cluster sob demanda | Grátis, "dorme" após 15 min |
| Complexidade operacional | Média (orquestração de job) | Baixa (um container) |
| Quando usar | Grandes volumes, sem urgência | Uso interativo, tempo real |

**Decisões que se repetem nos 2 deploys, por design:**

- **Pré-processamento vive dentro do `.pkl`** — o pipeline salvo contém o
  `StandardScaler`/`TargetEncoder` junto com o classificador, então nenhum
  deploy reimplementa encoding manualmente. Evita *training/serving skew*:
  qualquer divergência de implementação do pré-processamento faria o modelo
  receber dados fora da distribuição que aprendeu, gerando previsões
  erradas sem erro aparente.
- **Versões de bibliotecas fixadas** exatamente como no treino (Python
  3.10.18, scikit-learn 1.7.1, lightgbm 4.6.0) : divergência causa falha
  silenciosa só na hora do `joblib.load`, já em produção.
- **`joblib.load` no escopo do módulo**, carregado uma única vez na
  inicialização, nunca a cada requisição.
- **Configuração via ambiente** (`.env`, `$PORT`, `API_URL`), nunca
  hardcoded no código.

---

## 3. Stack

**Modelagem:** Python 3.10 · scikit-learn · LightGBM · pandas · sweetviz

**Deploy 1 (Batch):** Databricks · Spark · PostgreSQL

**Deploy 2 (API + Interface):** FastAPI · Streamlit · Docker · Render

---

## 4. Implementação

### Modelagem

| Etapa | O que faz |
|---|---|
| Split (holdout) | 20% dos dados isolados desde o início, nunca tocados até a validação final |
| EDA | Auditoria de tipos/missing, sweetviz, sem uso para seleção de features |
| Comparação de baselines | Regressão Logística, HistGradientBoosting, MLP : venceu Regressão Logística |
| Comparação de encoders | TargetEncoder, OneHotEncoder, OrdinalEncoder : empate por baixa cardinalidade |
| Tuning | RandomizedSearchCV (20 combinações, 5 folds) |
| Threshold | Testado de 0.3 a 0.7 : não afeta ROC AUC/log loss, só precision/recall/F1 |
| Validação final | Re-treino em treino+teste, avaliado no holdout intocado |
| Exportação | Pipeline completo salvo via joblib, com sanity check (`np.allclose`) pós-recarga |

Dois classificadores foram comparados: o modelo em produção usa
**LightGBM**; o notebook de referência (`00_modelagem/`) usa
**HistGradientBoostingClassifier**. Ambos os `.pkl` estão preservados no
repositório — detalhamento em [`models/README.md`](models/README.md).

### Deploy 1: Batch Agendado (Databricks)

Job que lê `consumer_shopping_input` no Postgres, roda `predict` com o
modelo carregado uma vez no escopo do módulo, e grava o resultado em
`shopping_preference_predictions`. **Validado com 157.100 registros reais.**

![Setup do notebook de inferência](docs/images/deploy1-notebook-setup.png)
![Pipeline de inferência](docs/images/deploy1-notebook-inferencia.png)

```sql
SELECT * FROM shopping_preference_predictions LIMIT 10;
```

| customer_id | batch_id | prediction | label | probability_online |
|---|---|---|---|---|
| c243005b-... | batch_20260904_174632 | 1 | Online | 0.9534041 |
| ae63293a-... | batch_20260904_174632 | 1 | Online | 0.999998 |
| 3d0d57cd-... | batch_20260904_174632 | 1 | Online | 0.9998865 |
| da5d8aba-... | batch_20260904_174632 | 0 | Store | 0.33768138 |
| 6288f6d8-... | batch_20260904_174632 | 0 | Store | 0.000000002 |
| 4a0c74e1-... | batch_20260904_174632 | 0 | Store | 0 |
| f9a2b79d-... | batch_20260904_174632 | 0 | Store | 0.000000447 |
| e228d624-... | batch_20260904_174632 | 0 | Store | 0.000000111 |
| da3d5d73-... | batch_20260904_174632 | 1 | Online | 0.883311 |
| 9a4ca7a5-... | batch_20260904_174632 | 1 | Online | 0.8880869 |

![Query no Postgres via DBeaver](docs/images/deploy1-query-postgres.png)

**Trade-off : prototipagem local vs. produção no Databricks:**

| | VS Code (local) | Databricks (produção) |
|---|---|---|
| Onde executa | Máquina local, `.venv` | Cluster remoto gerenciado |
| Segredos | `.env` + `python-dotenv` | Databricks Secrets (recomendado) |
| Acesso ao Postgres | Driver direto (`psycopg2`) | Leitura via Spark (`spark.read.format("postgresql")`) |

Usar `psycopg2` direto no ambiente Serverless do Databricks causa falha de
baixo nível (`SIGABRT`, sem traceback útil) - o runtime não oferece o mesmo
ambiente de sistema de uma máquina local. Corrigido usando leitura nativa
do Spark.

### Deploy 2 — API + Interface em Container (Docker → Render)

FastAPI serve o modelo via `/predict`; Streamlit consome essa API via HTTP,
com os 24 campos organizados em 3 seções. Empacotado em uma imagem Docker
(`python:3.10-slim`), publicada no Render.

![Resultado da previsão](docs/images/deploy2-resultado.png)

**Imprevisto : porta fixa vs. porta dinâmica do Render:** primeiro deploy
resultou em "Not Found". Causa: o Dockerfile fixava a porta do Streamlit em
`8501`, mas o Render atribui a porta via `$PORT` e só roteia tráfego para
ela. Corrigido trocando a porta fixa por `${PORT}` no `CMD`.

**Imprevisto : `libgomp.so.1` ausente na imagem slim:** o LightGBM depende
de OpenMP, não incluído por padrão em `python:3.10-slim`. Resolvido
instalando `libgomp1` via `apt-get` antes das dependências Python.

![Histórico de deploys no Render](docs/images/deploy2-render-deploy.png)

---

## 5. Resultados

**Resultados:**
- Deploy 1 processou 157.100 registros reais em lote, sem erro
- Deploy 2 está no ar publicamente, respondendo em tempo real
- Mesmo `.pkl`, mesmo contrato de pré-processamento, dois padrões de
  arquitetura diferentes sem duplicar lógica de inferência

