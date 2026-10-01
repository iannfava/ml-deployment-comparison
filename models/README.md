# Modelos

Esta pasta guarda o pipeline gerado pelo notebook de modelagem
(`00_modelagem/modelo_aula.ipynb`), mantido como referência. **Não é o
modelo usado em produção.**

## Qual modelo está em produção

O modelo usado nos Deploys 1 e 2 (`shopping_preference_model.pkl`, presente
em `01_deploy_batch_databricks/` e `02_deploy_api_container/`) usa
**LightGBM** (`LGBMClassifier`) como classificador final e foi salvo com
scikit-learn 1.7.1. Ele veio pronto, sem notebook de treino.

## Sobre este arquivo (`shopping_preference_pipeline.pkl`)

Este pipeline usa **HistGradientBoostingClassifier**, a implementação nativa
de gradient boosting do scikit-learn, e foi salvo com scikit-learn 1.9.0,
uma versão diferente da fixada no restante do projeto.

| | Produção (`shopping_preference_model.pkl`) | Este arquivo (`shopping_preference_pipeline.pkl`) |
|---|---|---|
| Classificador | LightGBM (`LGBMClassifier`) | scikit-learn nativo (`HistGradientBoostingClassifier`) |
| scikit-learn (salvo com) | 1.7.1 | 1.9.0 |
| Features de entrada | 24 | 24 |
| Classes (`classes_`) | 0 e 1 (o código dos deploys trata 1 como Online) | `'Online'`, `'Store'` (texto, em ordem alfabética) |
| Uso | Deploys 1 e 2 | Só referência, nunca implantado |

## ⚠️ Atenção: ordem das classes

Este pipeline foi treinado com o alvo em texto, então
`predict_proba(X)[:, 1]` devolve a probabilidade de **Store**, e não de
Online. Para obter a probabilidade de Online, use a coluna 0, ou confira
sempre `classes_` antes:

```python
online = list(modelo.classes_).index("Online")
proba_online = modelo.predict_proba(X)[:, online]
```

Por esse motivo, a célula de validação final do notebook, que usa `[:, 1]`,
mostra métricas invertidas (acurácia de 1,65%). As métricas do conjunto de
teste, calculadas antes no mesmo notebook, não têm esse problema.

## Compatibilidade

**Não use este arquivo nos deploys** sem antes ajustar a versão: as versões
fixadas do projeto (scikit-learn 1.7.1, lightgbm 4.6.0) não correspondem às
usadas para salvar este pipeline.
