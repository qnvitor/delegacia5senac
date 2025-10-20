# ==========================================================
# Gráficos e análises exploratórias
# ==========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")


def plot_exploratory(df: pd.DataFrame):
    """Exploração visual dos dados principais."""

    # Crimes mais frequentes
    plt.figure(figsize=(10,5))
    sns.countplot(y=df["tipo_crime"], order=df["tipo_crime"].value_counts().index[:10])
    plt.title("Top 10 crimes mais cometidos")
    plt.show()

    # Bairros
    plt.figure(figsize=(14,6))
    top_bairros = df["bairro"].value_counts().index[:10]
    sns.countplot(data=df[df["bairro"].isin(top_bairros)], x="bairro", hue="tipo_crime")
    plt.xticks(rotation=45)
    plt.title("Distribuição dos tipos de crime por bairro (Top 10)")
    plt.tight_layout()
    plt.show()

    # Armas
    plt.figure(figsize=(14,6))
    sns.countplot(data=df, x="tipo_crime", hue="arma_utilizada", order=df["tipo_crime"])
    plt.xticks(rotation=45)
    plt.title("Uso de armas por tipo de crime")
    plt.tight_layout()
    plt.show()

    # Idade dos suspeitos
    plt.figure(figsize=(14,6))
    sns.boxplot(data=df, x="tipo_crime", y="idade_suspeito", order=df["tipo_crime"])
    plt.xticks(rotation=45)
    plt.title("Distribuição da idade dos suspeitos por tipo de crime")
    plt.tight_layout()
    plt.show()
