# Modelos

Esta pasta guarda o resultado direto do notebook de modelagem
(`00_modelagem/modelo_aula.ipynb`), mantido aqui como registro da comparação
de classificadores feita durante a modelagem — **não é o modelo usado em
produção**.

## Qual modelo está em produção

O modelo usado nos Deploys 1 e 2 (`shopping_preference_model.pkl`, presente
em `01_deploy_batch_databricks/` e `02_deploy_api_container/`) usa
**LightGBM** (`LGBMClassifier`) como classificador final, treinado com
scikit-learn 1.7.1.

## Sobre este arquivo (`shopping_preference_pipeline.pkl`)

Este pipeline usa **HistGradientBoostingClassifier**, a implementação nativa
de gradient boosting do scikit-learn, treinado com scikit-learn 1.9.0 — uma
versão diferente da fixada para o restante do projeto.

| | Produção (`shopping_preference_model.pkl`) | Este arquivo (`shopping_preference_pipeline.pkl`) |
|---|---|---|
| Classificador | LightGBM (`LGBMClassifier`) | scikit-learn nativo (`HistGradientBoostingClassifier`) |
| scikit-learn (treino) | 1.7.1 | 1.9.0 |
| Features | 24 | 24 |
| Uso | Deploy 1 e Deploy 2, validado com dados reais | Referência da modelagem, não deployado |

## Por que os dois existem

São duas famílias de gradient boosting testadas durante a comparação de
modelos: LightGBM (biblioteca externa, otimizada para velocidade em grandes
volumes) e HistGradientBoosting (implementação nativa do scikit-learn,
mesma ideia, sem dependência externa). O material de referência mais
completo do curso usa HistGradientBoosting, mas o modelo efetivamente
colocado em produção pelo material original usa LightGBM.

**Não use este arquivo em nenhum dos deploys** sem antes confirmar
compatibilidade de versão — as versões fixadas do projeto (scikit-learn
1.7.1, lightgbm 4.6.0) não correspondem às usadas para treinar este pipeline.
