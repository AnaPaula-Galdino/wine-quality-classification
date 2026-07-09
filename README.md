# Wine Quality Classification — Tech Challenge Fase 2

Classificação binária da qualidade de vinhos a partir de características físico-químicas, desenvolvida para o **Tech Challenge da Fase 2 — POSTECH (Data Analytics / DTAT)**.

**Autora:** Ana Paula Corrêa Galdino — RM 370461

**Repositório:** `<INSERIR LINK DO GITHUB APÓS A PUBLICAÇÃO>`

## O problema

A avaliação de qualidade de vinhos é feita por especialistas via análise sensorial, um processo subjetivo, caro e demorado. Este projeto usa dados físico-químicos medidos rotineiramente em laboratório para prever se um vinho é de **Alta Qualidade (nota ≥ 7)** ou **Baixa/Média Qualidade (nota < 7)**, permitindo triagem automática de lotes e ajustes no processo produtivo.

**Fonte dos dados:** [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (`WineQT.csv`, 1.143 amostras de vinho tinto, 11 variáveis físico-químicas e a nota de qualidade).

## Principais resultados

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,789 | 0,357 | 0,741 | 0,482 | 0,890 |
| **Random Forest** (selecionado) | **0,902** | **0,621** | **0,667** | **0,643** | **0,918** |
| XGBoost | 0,907 | 0,667 | 0,593 | 0,627 | 0,902 |

Em validação cruzada os três modelos ficam tecnicamente empatados; o **Random Forest** foi selecionado por apresentar o melhor conjunto de métricas no teste (F1, ROC AUC e Average Precision de 0,732, contra 0,595 do XGBoost). A incerteza da comparação foi quantificada com teste de McNemar e intervalo de confiança bootstrap, documentados no notebook. Apenas 13,5% dos vinhos são de alta qualidade, e esse desbalanceamento orientou todas as decisões de metodologia.

**Variáveis mais influentes** (confirmadas por importância de impureza e por permutação): teor alcoólico, interação álcool × sulfatos (feature criada no projeto), acidez volátil e sulfatos, todas com fundamento enológico.

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/                  # Base de dados (WineQT.csv)
├── notebooks/             # Notebook completo com análise e modelagem
├── src/                   # Scripts auxiliares (pré-processamento e modelagem)
├── results/               # Gráficos e métricas dos modelos
├── docs/                  # Relatório executivo e apresentação
├── requirements.txt       # Bibliotecas utilizadas (versões fixadas)
└── README.md
```

O notebook é autocontido de propósito: o avaliador executa um único arquivo do início ao fim. A pasta `src/` oferece a mesma lógica em formato modular, para reuso via linha de comando.

## Como reproduzir

```bash
# 1. Clonar o repositório e instalar as dependências
pip install -r requirements.txt

# 2. Executar o notebook completo (recomendado)
jupyter notebook notebooks/wine_quality_classification.ipynb

# 3. Ou executar apenas o pipeline de modelagem via linha de comando
python src/modeling.py
```

Todas as etapas estocásticas usam `random_state=42`; os resultados são totalmente reprodutíveis.

## Metodologia (resumo)

1. **Compreensão do problema**: binarização da nota (≥ 7 = alta qualidade), justificada pela decisão de negócio e pela escassez das classes extremas.
2. **EDA**: distribuições, boxplots e violin plots por classe, matriz de correlação comentada (Pearson e Spearman), testes de Mann-Whitney por variável, análise de outliers (IQR) e do balanceamento de classes.
3. **Pré-processamento**: remoção de 125 duplicatas (prevenção de data leakage), sem imputação (zero faltantes), padronização via `Pipeline` aplicada apenas à Regressão Logística (árvores dispensam escala), feature engineering validada por ganho de F1 em validação cruzada.
4. **Modelagem**: Logistic Regression (baseline), Random Forest e XGBoost, com `GridSearchCV`, validação cruzada estratificada de 5 dobras e ponderação de classes.
5. **Avaliação**: acurácia, precisão, recall, F1, ROC AUC, curvas ROC e Precision-Recall, matriz de confusão, teste de McNemar e IC bootstrap do F1.
6. **Interpretação**: importância das features (impureza e permutação) e implicações para o processo produtivo.

## Entregáveis

- Notebook: [`notebooks/wine_quality_classification.ipynb`](notebooks/wine_quality_classification.ipynb)
- Apresentação executiva (storytelling da EDA): [`docs/`](docs/)
- Relatório executivo: [`docs/`](docs/)
- Gráficos e métricas: [`results/`](results/)
