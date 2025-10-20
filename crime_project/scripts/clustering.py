# ==========================================================
# Clusterização e interpretação
# ==========================================================
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("Set2")

def embed_and_reduce(X, n_components=2, method="umap"):
    """Redução de dimensionalidade para visualização (UMAP ou PCA)."""
    if method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    emb = reducer.fit_transform(X)
    return emb

def run_kmeans(X, n_clusters=5, use_text_col=None):
    """Executa KMeans e retorna labels e score. Pode incluir TF-IDF de texto."""
    X_proc = X.copy()
    if use_text_col and use_text_col in X_proc.columns:
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=2)
        X_text = tfidf.fit_transform(X_proc[use_text_col].fillna("")).toarray()
        X_proc = pd.concat([X_proc.drop(columns=[use_text_col]), pd.DataFrame(X_text)], axis=1)

    print(f"🔷 Rodando KMeans com k={n_clusters}")
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_proc)
    score = silhouette_score(X_proc, labels) if len(set(labels)) > 1 else -1
    print(f"Silhouette Score (KMeans): {score:.3f}")
    return model, labels, score

def run_hdbscan(X, min_cluster_size=15, metric="euclidean", use_text_col=None):
    """Executa HDBSCAN — bom para clusters com densidade variável. Pode incluir TF-IDF."""
    X_proc = X.copy()
    if use_text_col and use_text_col in X_proc.columns:
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=2)
        X_text = tfidf.fit_transform(X_proc[use_text_col].fillna("")).toarray()
        X_proc = pd.concat([X_proc.drop(columns=[use_text_col]), pd.DataFrame(X_text)], axis=1)

    print(f"🔷 Rodando HDBSCAN (min_cluster_size={min_cluster_size})")
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = model.fit_predict(X_proc)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"HDBSCAN clusters encontrados (excluindo -1): {n_clusters}")
    return model, labels

def describe_clusters(df_orig, labels, topk=5):
    """Gera descrições simples por cluster com count das features mais relevantes."""
    df = df_orig.copy()
    df["cluster"] = labels
    descriptions = {}
    for c in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == c]
        desc = {}
        desc["count"] = len(subset)
        for col in ["bairro", "tipo_crime", "arma_utilizada"]:
            if col in df.columns:
                desc[f"top_{col}"] = subset[col].value_counts().head(topk).to_dict()
        descriptions[c] = desc
    return descriptions

def plot_clusters(emb, labels, title="Clusters"):
    """Scatter plot dos embeddings coloridos por cluster. Retorna a figura."""
    fig, ax = plt.subplots(figsize=(8,6))
    unique = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique))
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(emb[mask,0], emb[mask,1], label=str(lab), alpha=0.6, s=8)
    ax.legend(title="Cluster")
    ax.set_title(title)
    fig.tight_layout()
    return fig
