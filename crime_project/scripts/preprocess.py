# ==========================================================
# preprocess.py
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================
# Função global para extrair coluna de texto
# ============================
def get_text_column(X, col_name):
    """Extrai a coluna de texto como vetor 1D de strings para TF-IDF."""
    return X[col_name].astype(str)

# ============================
# Tratamento de outliers numéricos
# ============================
def handle_outliers(df, numericas, method="clip"):
    """
    Trata outliers em colunas numéricas.
    method: 'clip' (limita pelo IQR) ou 'drop' (remove linhas)
    """
    df = df.copy()
    for col in numericas:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if method == "clip":
            df.loc[:, col] = df[col].clip(lower, upper)
        elif method == "drop":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# ============================
# Preparar features do dataset
# ============================
def prepare_features(df):
    """
    Retorna df_modelo, X_all e y.
    Assume que 'tipo_crime' é target.
    Extrai componentes de data da coluna 'data_ocorrencia'.
    """
    df_modelo = df.copy()

    # Extrair componentes de data se a coluna existir
    if "data_ocorrencia" in df_modelo.columns:
        df_modelo["data_ocorrencia"] = pd.to_datetime(df_modelo["data_ocorrencia"])
        df_modelo["ano"] = df_modelo["data_ocorrencia"].dt.year
        df_modelo["mes"] = df_modelo["data_ocorrencia"].dt.month
        df_modelo["dia"] = df_modelo["data_ocorrencia"].dt.day
        df_modelo["dia_semana"] = df_modelo["data_ocorrencia"].dt.dayofweek
        
        # Remover a coluna original de data após extrair os componentes
        df_modelo = df_modelo.drop(columns=["data_ocorrencia"])

    # Target
    if "tipo_crime" in df_modelo.columns:
        y = df_modelo["tipo_crime"]
    else:
        y = None

    # Features principais
    X_all = df_modelo.copy()
    if "tipo_crime" in X_all.columns:
        X_all = X_all.drop(columns=["tipo_crime"])

    return df_modelo, X_all, y

# ============================
# Construir preprocessor completo
# ============================
def build_preprocessor(categoricas, numericas, text_cols=None):
    """
    Cria ColumnTransformer completo:
    - Categóricas → imputação + OneHotEncoder
    - Numéricas → imputação + StandardScaler
    - Texto → TF-IDF
    """
    transformers = []

    # ============================
    # CATEGÓRICAS
    # ============================
    if categoricas:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, categoricas))

    # ============================
    # NUMÉRICAS
    # ============================
    if numericas:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, numericas))

    # ============================
    # TEXTO (TF-IDF)
    # ============================
    if text_cols:
        for col in text_cols:
            text_pipe = Pipeline([
                ("selector", FunctionTransformer(func=get_text_column, kw_args={"col_name": col}, validate=False)),
                ("tfidf", TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1,2),
                    min_df=2,
                    max_df=0.95
                ))
            ])
            transformers.append((f"tfidf_{col}", text_pipe, [col]))

    # ============================
    # COMBINA TUDO
    # ============================
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor
