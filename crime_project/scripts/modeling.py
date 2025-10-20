# ==========================================================
# Modelagem, avaliação e interpretação
# ==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, f1_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def train_models(X_train, y_train, X_test, y_test, preprocessador):
    """Treina modelos (Dummy, LR, RF) e retorna resultados."""
    modelos = {}

    # Baseline
    dummy = ImbPipeline([("prep", preprocessador), ("clf", DummyClassifier(strategy="most_frequent"))])
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    print("Baseline DummyClassifier:\n", classification_report(y_test, y_pred_dummy))
    modelos["Dummy"] = (dummy, y_pred_dummy)

    # Logistic Regression
    lr = ImbPipeline([("prep", preprocessador), ("smote", SMOTE(random_state=42)),
                      ("clf", LogisticRegression(max_iter=500))])
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Regressão Logística:\n", classification_report(y_test, y_pred_lr))
    modelos["Logistic Regression"] = (lr, y_pred_lr)

    # Random Forest
    rf = ImbPipeline([("prep", preprocessador), ("smote", SMOTE(random_state=42)),
                      ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest:\n", classification_report(y_test, y_pred_rf))
    modelos["Random Forest"] = (rf, y_pred_rf)

    return modelos


def compare_results(modelos, y_test):
    """Compara métricas dos modelos."""
    resultados = []
    for nome, (modelo, y_pred) in modelos.items():
        resultados.append({
            "Modelo": nome,
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados)
    return df_resultados


def plot_confusion_matrix(modelo_rf, y_test, y_pred_rf):
    """Exibe matriz de confusão normalizada."""
    cm = confusion_matrix(y_test, y_pred_rf, labels=modelo_rf.classes_, normalize="true")
    cm_df = pd.DataFrame(cm, index=modelo_rf.classes_, columns=modelo_rf.classes_)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Matriz de Confusão Normalizada - Random Forest")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.show()
