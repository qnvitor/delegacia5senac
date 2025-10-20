# ==========================================================
# Detecção de anomalias
# ==========================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def isolation_forest_detect(X, contamination=0.02):
    """Detecta anomalias com IsolationForest. Usa apenas colunas numéricas."""
    X_num = X.select_dtypes(include=[np.number]).copy()
    print(f"⚠️ Rodando IsolationForest (contamination={contamination}) sobre {X_num.shape[1]} features numéricas")
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X_num)
    scores = -model.decision_function(X_num)  # maior = mais anômalo
    is_outlier = preds == -1
    return model, is_outlier, scores

def lof_detect(X, n_neighbors=20, contamination=0.02):
    """Local Outlier Factor — retorna boolean mask (True=outlier) e scores. Usa apenas numéricas."""
    X_num = X.select_dtypes(include=[np.number]).copy()
    print(f"⚠️ Rodando LOF (n_neighbors={n_neighbors}, contamination={contamination}) sobre {X_num.shape[1]} features numéricas")
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    preds = model.fit_predict(X_num)
    scores = -model.negative_outlier_factor_  # maior = mais anômalo
    is_outlier = preds == -1
    return model, is_outlier, scores
