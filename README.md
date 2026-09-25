# Credit Scoring — Construção de Target e Modelo de Classificação de Risco

Projeto de portfólio em Ciência de Dados: construção de um modelo de classificação de risco de crédito a partir de uma base **sem variável target pronta**, exigindo a definição do rótulo "bom"/"ruim" cliente por meio de análise de vintage antes de qualquer etapa de modelagem.

## Contexto do problema

Sistemas de pontuação de crédito usam dados históricos para prever a probabilidade de inadimplência de um solicitante. Neste dataset, ao contrário de tarefas de classificação convencionais, não existe uma coluna de rótulo pronta — é necessário construí-la a partir do histórico de pagamentos, um desafio real e comum em problemas de risco de crédito no mercado.

## Dados

- **`application_record.csv`** — características dos clientes no momento da solicitação (renda, idade, ocupação, composição familiar, etc.)
- **`credit_record.csv`** — histórico mensal de status de pagamento por cliente, usado para construir o target

Fonte: [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) (Kaggle).

## Metodologia

### 1. Construção do target (Vintage Analysis)

Sem rótulo pronto, a definição de cliente "bad" exigiu duas decisões, cada uma testada com evidência em vez de assumida:

- **Janela mínima de observação**: 36 meses — escolhida analisando o trade-off entre confiabilidade do rótulo (quanto mais tempo observado, mais o comportamento real se revela) e tamanho de amostra retida.
- **Definição de inadimplência**: atraso superior a 60 dias.

Essa combinação capturou **~61% dos eventos de inadimplência** que ocorrem ao longo da vida da conta, preservando **~21% da base elegível** para modelagem — uma limitação conhecida e documentada, não escondida do resultado final.

### 2. Análise Exploratória (EDA)

Principais achados:

- Desbalanceamento severo do target: **~4,4%** dos clientes elegíveis são "bad".
- Clientes mais jovens, com menor tempo de emprego, viúvos e pensionistas apresentaram taxa de inadimplência acima da média da carteira.
- O caso dos pensionistas foi investigado por hipótese (maior exposição temporal da conta) e **refutado com dados** — a diferença observada não foi explicada pela hipótese de maior tempo de observação.
- Identificação e tratamento de um valor sentinela (365243 dias) em `DAYS_EMPLOYED`, usado para representar clientes sem ocupação atual — sem esse tratamento, a comparação de tempo de emprego entre clientes bons e ruins saía invertida.
- Investigação da causa estrutural de nulos em `OCCUPATION_TYPE`: 99,8% dos pensionistas não têm ocupação informada (esperado, por não estarem mais no mercado de trabalho), enquanto ~16-17% dos clientes economicamente ativos têm o mesmo campo nulo sem explicação estrutural — tratados como grupos distintos.

### 3. Pré-processamento

- Codificação de variáveis binárias e one-hot encoding das variáveis categóricas nominais, dentro de um `ColumnTransformer` (fit apenas no conjunto de treino, evitando vazamento de dados).
- Imputação de nulos com significado semântico preservado (ex.: tempo de emprego "não aplicável" tratado com flag dedicada, não com média/mediana).
- Split treino/teste estratificado, preservando a proporção de 4,4% de "bad" em ambos os conjuntos.

### 4. Modelagem

Dois algoritmos testados — **Random Forest** e **Gradient Boosting** — com tratamento de desbalanceamento (`class_weight` e `sample_weight`, respectivamente) e refinamento de hiperparâmetros via `RandomizedSearchCV` com validação cruzada estratificada, otimizando PR-AUC (métrica apropriada para classes raras, ao contrário de acurácia).

## Resultados

| Modelo                               | Precision (bad) | Recall (bad) | F1 (bad) | ROC-AUC | PR-AUC |
| ------------------------------------ | --------------- | ------------ | -------- | ------- | ------ |
| Random Forest — antes do tuning     | 0,50            | 0,36         | 0,42     | 0,72    | 0,31   |
| Random Forest — refinado            | 0,46            | 0,36         | 0,40     | 0,71    | 0,33   |
| Gradient Boosting — antes do tuning | 0,09            | 0,46         | 0,15     | 0,65    | 0,13   |
| Gradient Boosting — refinado        | 0,33            | 0,39         | 0,36     | 0,69    | 0,31   |

No threshold padrão de 0,5, o Random Forest apresentou Precision e F1-score superiores para a classe bad, enquanto o Gradient Boosting apresentou Recall ligeiramente maior. Mas o achado mais relevante do tuning não foi declarar um vencedor: o Gradient Boosting, inicialmente sub-otimizado (poucos estimadores com learning rate baixo), fechou quase toda a diferença de capacidade discriminativa após o ajuste — o PR-AUC saltou de 0,13 para 0,31, ficando próximo dos 0,33 do Random Forest.

## Limitações

- Mesmo após o refinamento, o recall da classe "bad" permanece em ~36-39% — uma parcela relevante dos clientes que de fato ficam inadimplentes ainda não é identificada no threshold padrão.
- ~40% dos eventos de inadimplência futuros não estão capturados pela janela de observação escolhida, uma limitação aceita em troca de manter tamanho de amostra viável.
- Categorias com poucas observações (ex.: nível educacional "Academic degree", com 8 clientes) não sustentam conclusões estatísticas robustas.

## Próximos passos

- Ajustar o threshold de decisão considerando o custo de negócio entre falso positivo (negar bom cliente) e falso negativo (aprovar cliente que será inadimplente).
- Testar estratégias adicionais de balanceamento (ex.: SMOTE).
- Avaliar se uma janela de observação mais longa, mesmo reduzindo ainda mais a amostra, produziria um target mais confiável.

## Stack

Python · pandas · scikit-learn · matplotlib / seaborn

