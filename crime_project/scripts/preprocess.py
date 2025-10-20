# ==========================================================
# Limpeza e preparação dos dados
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")


def prepare_features(df: pd.DataFrame):
    """Cria colunas derivadas da data e separa features/alvo."""
    df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"], errors="coerce")
    df["mes"] = df["data_ocorrencia"].dt.month
    df["dia"] = df["data_ocorrencia"].dt.day
    df["ano"] = df["data_ocorrencia"].dt.year
    df["dia_semana"] = df["data_ocorrencia"].dt.dayofweek

    colunas_features = [
        "bairro",
        "descricao_modus_operandi",
        "arma_utilizada",
        "quantidade_vitimas",
        "quantidade_suspeitos",
        "sexo_suspeito",
        "idade_suspeito",
        "mes",
        "dia",
        "ano",
        "dia_semana",
    ]
    coluna_alvo = "tipo_crime"

    df_modelo = df[colunas_features + [coluna_alvo]].copy()
    X = df_modelo[colunas_features]
    y = df_modelo[coluna_alvo]
    return df_modelo, X, y


def handle_outliers(X: pd.DataFrame, numericas: list):
    """Trata outliers com método IQR (clipping)."""
    for col in numericas:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        X[col] = np.where(X[col] < limite_inferior, limite_inferior,
                 np.where(X[col] > limite_superior, limite_superior, X[col]))
    return X


def build_preprocessor(categoricas, numericas):
    """Cria o pré-processador com OneHotEncoder + StandardScaler."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", StandardScaler(), numericas)
    ])
