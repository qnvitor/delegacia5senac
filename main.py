import os
import pandas as pd
import joblib
from scripts.utils import ensure_outputs, load_dataset, check_columns
from scripts.preprocess import prepare_features, handle_outliers, build_preprocessor
from scripts.visualization import plot_exploratory, heatmap_hotspots
from scripts.clustering import run_kmeans, embed_and_reduce, describe_clusters
from scripts.anomaly_detection import isolation_forest_detect
from scripts.modeling import train_classifiers
from sklearn.model_selection import train_test_split

def run():
    """
    Função principal para preparar os dados e treinar os modelos.
    Pode ser chamada tanto pelo Streamlit quanto diretamente pelo terminal.
    """
    print("✅ Iniciando pré-processamento e treinamento...")
    ensure_outputs()

    path = "data/dataset_ocorrencias_delegacia_5.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Coloque o CSV em data/dataset_ocorrencias_delegacia_5.csv")

    df = load_dataset(path)
    req_cols = [
        "data_ocorrencia",
        "tipo_crime",
        "latitude",
        "longitude",
        "bairro",
        "descricao_modus_operandi"
    ]
    check_columns(df, required=req_cols)

    plot_exploratory(df, save=True)
    heatmap_hotspots(df)


    df_modelo, X_all, y = prepare_features(df)

    # Identificar tipos de colunas
    numericas = [
        c for c in [
            "quantidade_vitimas", "quantidade_suspeitos", "idade_suspeito",
            "mes", "dia", "ano", "dia_semana"
        ] if c in X_all.columns
    ]
    categoricas = [c for c in ["bairro", "arma_utilizada", "sexo_suspeito"] if c in X_all.columns]
    text_cols = ["descricao_modus_operandi"] if "descricao_modus_operandi" in X_all.columns else []


    if numericas:
        X_all.loc[:, numericas] = handle_outliers(X_all, numericas)

    preproc = build_preprocessor(categoricas, numericas, text_cols=text_cols)
    preproc.fit(X_all, y)
    joblib.dump(preproc, "outputs/preprocessor_full.pkl")
    print("✅ Preprocessor (com TF-IDF) fitado e salvo em outputs/preprocessor_full.pkl")

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train = X_test = y_train = y_test = None


    if y is not None:
        results = train_classifiers(X_train, y_train, X_test, y_test, preproc)


        cols_for_clust = numericas + categoricas
        X_clust = pd.get_dummies(
            X_all[cols_for_clust].copy().fillna("NA"),
            drop_first=True
        )

        model_k, labels_k, score_k = run_kmeans(X_clust, n_clusters=4)
        emb = embed_and_reduce(X_clust)
        desc = describe_clusters(df, labels_k)

        print("📊 Descrições de clusters:")
        print(desc)
    else:
        print("⚠️ Sem target — pulando modelagem supervisionada.")

if __name__ == "__main__":
    run()
