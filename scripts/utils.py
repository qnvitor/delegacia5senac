# ==========================================================
# Funções auxiliares e checagens
# ==========================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib

def ensure_outputs(path="outputs"):
    """Garante que a pasta outputs/ exista."""
    os.makedirs(path, exist_ok=True)
    return path

def load_dataset(path: str) -> pd.DataFrame:
    """Carrega o dataset principal com mensagens explicativas."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    print(f"✅ Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df

def safe_head(df: pd.DataFrame, n=5):
    """Mostra as primeiras linhas de forma segura."""
    print(df.head(n))

def save_model(model, path=None):
    """Salva qualquer modelo ou objeto em outputs/."""
    ensure_outputs()
    if path is None:
        # Salvamento padrão para pipeline completo ou modelo
        if hasattr(model, "predict"):
            path = "outputs/rf_model.pkl"
        else:
            path = "outputs/preprocessor_full.pkl"
    joblib.dump(model, path)
    print(f"💾 Objeto salvo em {path}")

def load_model(path):
    """Carrega um modelo ou objeto salvo em outputs/."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    model = joblib.load(path)
    print(f"✅ Objeto carregado de {path}")
    return model

def check_columns(df: pd.DataFrame, required=[]):
    """Verifica colunas obrigatórias e retorna dict de flags."""
    present = {col: (col in df.columns) for col in required}
    for col, ok in present.items():
        if not ok:
            warnings.warn(
                f"Atenção: coluna '{col}' não encontrada no dataset. Algumas funcionalidades serão ignoradas."
            )
    return present

def sample_for_shap(X: pd.DataFrame, frac=0.2, random_state=42) -> pd.DataFrame:
    """Amostra X para SHAP (protege contra grandes datasets)."""
    if X.shape[0] > 2000:
        return X.sample(frac=frac, random_state=random_state)
    return X.copy()
