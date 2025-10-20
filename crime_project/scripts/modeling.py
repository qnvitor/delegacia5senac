# ==========================================================
# Modelagem, avaliação, tunagem, fairness e SHAP
# ==========================================================
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier, Pool
import joblib
from scripts.utils import save_model, sample_for_shap

def train_classifiers(X_train, y_train, X_test, y_test, preprocessor, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # Logistic Regression simplificada
    from sklearn.linear_model import LogisticRegression
    print("🔷 Treinando Regressão Logística com SMOTE...")
    lr_pipe = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=500))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    print("Regressão Logística:\n", classification_report(y_test, y_pred_lr))
    joblib.dump(lr_pipe, os.path.join(save_dir, "logistic_model.pkl"))
    results["logistic"] = (lr_pipe, y_pred_lr)

    # CatBoost Classifier
    print("🔷 Treinando CatBoostClassifier com SMOTE...")
    cb_pipe = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", CatBoostClassifier(iterations=500, learning_rate=0.1,
                                   depth=6, eval_metric="Accuracy",
                                   verbose=100, random_seed=42))
    ])
    cb_pipe.fit(X_train, y_train)
    y_pred_cb = cb_pipe.predict(X_test)
    print("CatBoostClassifier:\n", classification_report(y_test, y_pred_cb))
    joblib.dump(cb_pipe, os.path.join(save_dir, "cb_model.pkl"))
    results["catboost"] = (cb_pipe, y_pred_cb)

    return results
