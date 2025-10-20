# ==========================================================
# Funções auxiliares
# ==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(path: str):
    """Carrega o dataset principal."""
    df = pd.read_csv(path)
    print(f"✅ Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df


def show_nulls(df: pd.DataFrame):
    """Exibe a contagem de valores nulos."""
    print("Valores nulos por coluna:")
    print(df.isnull().sum())


def plot_bar(df, x, y, title="", rotation=45):
    """Gera um gráfico de barras simples."""
    plt.figure(figsize=(10,5))
    sns.barplot(x=df[x], y=df[y])
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()
