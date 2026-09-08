Dia 1 — Notebook de Modelagem (EDA)
Erro ao rodar sv.analyze() no sweetviz:
  ValueError: Target feature 'shopping_preference' contains NaN (missing) values.
Causa: rodei a célula de "Tratamentos Iniciais" (.map({"Online": 1, "Store": 0})) duas vezes sem perceber.
1ª execução: coluna tinha texto → mapeou certo, virou 0/1.
2ª execução: coluna já era 0/1 (número) → o .map() procurou pelas chaves "Online"/"Store" (texto), não achou correspondência, e converteu toda a coluna para NaN (9431 valores nulos, ou seja, 100% das linhas).
Diagnóstico: comparei df_origem["shopping_preference"].unique() (só Store/Online, dado limpo) com df["shopping_preference"].unique() (retornou [nan]) — confirmou que o problema era de execução, não de dado sujo.
Correção: re-executei a célula do Split (recria df a partir de df_origem, com texto intacto) e depois rodei o .map() uma única vez.
Aprendizado: células que fazem transformação in-place (.map(), .drop(), .fillna(), etc.) não são idempotentes — rodar duas vezes muda o resultado silenciosamente, sem erro na hora. Se precisar re-testar algo no meio do notebook, é mais seguro reiniciar o kernel e rodar tudo em sequência (Run All) do que rodar células soltas fora de ordem.
Removi as células de diagnóstico (isnull().sum(), unique()) do notebook final depois de resolver — ficaram só aqui no diário.