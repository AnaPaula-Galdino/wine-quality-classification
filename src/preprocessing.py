"""Funções de pré-processamento do Tech Challenge Fase 2.

Pipeline: carga -> limpeza (duplicatas) -> alvo binário ->
feature engineering -> split estratificado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
QUALITY_THRESHOLD = 7
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WineQT.csv"


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Carrega a base, remove a coluna Id e as duplicatas físico-químicas.

    A remoção de duplicatas evita vazamento de dados entre treino e teste
    e o peso artificial de amostras repetidas.
    """
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df.drop_duplicates().reset_index(drop=True)


def add_binary_target(df: pd.DataFrame,
                      threshold: int = QUALITY_THRESHOLD) -> pd.DataFrame:
    """Cria a variável alvo binária: 1 se quality >= threshold (Alta Qualidade)."""
    df = df.copy()
    df["alta_qualidade"] = (df["quality"] >= threshold).astype(int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as features derivadas do projeto.

    A adoção destas features foi decidida no notebook, onde o ganho de
    F1 em validação cruzada foi medido (0,3775 -> 0,4076). Este módulo
    aplica a decisão já validada; a lógica do teste está documentada na
    Seção 4.1 do notebook.

    - razao_so2: fração ativa do conservante (SO2 livre / SO2 total);
    - interacao_alcool_sulfatos: combinação corpo + proteção antioxidante.
    """
    df = df.copy()
    df["razao_so2"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]
    df["interacao_alcool_sulfatos"] = df["alcohol"] * df["sulphates"]
    return df


def split_features_target(df: pd.DataFrame):
    """Separa features e alvo, aplicando o feature engineering."""
    features = [c for c in df.columns if c not in ("quality", "alta_qualidade")]
    x = add_engineered_features(df[features])
    y = df["alta_qualidade"]
    return x, y


def train_test_split_stratified(x, y, test_size: float = 0.2):
    """Split 80/20 estratificado e reprodutível (random_state=42)."""
    return train_test_split(x, y, test_size=test_size,
                            stratify=y, random_state=SEED)
