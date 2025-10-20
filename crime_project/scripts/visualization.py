# ==========================================================
# Gráficos, mapas e exportação
# ==========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from folium.plugins import HeatMap
import os

plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")

def plot_exploratory(df: pd.DataFrame, save=False):
    """Exploração visual dos dados principais. Salva arquivos em outputs/ se save=True."""
    os.makedirs("outputs", exist_ok=True)

    # Crimes mais frequentes
    if "tipo_crime" in df.columns:
        plt.figure(figsize=(10,5))
        sns.countplot(y=df["tipo_crime"], order=df["tipo_crime"].value_counts().index[:10])
        plt.title("Top 10 crimes mais cometidos")
        if save:
            plt.savefig("outputs/top10_crimes.png", bbox_inches="tight")
            print("📁 saved outputs/top10_crimes.png")

    # Bairros
    if "bairro" in df.columns and "tipo_crime" in df.columns:
        plt.figure(figsize=(14,6))
        top_bairros = df["bairro"].value_counts().index[:10]
        sns.countplot(data=df[df["bairro"].isin(top_bairros)], x="bairro", hue="tipo_crime")
        plt.xticks(rotation=45)
        plt.title("Distribuição dos tipos de crime por bairro (Top 10)")
        plt.tight_layout()
        if save:
            plt.savefig("outputs/tipo_por_bairro.png", bbox_inches="tight")
            print("📁 saved outputs/tipo_por_bairro.png")

    # Armas
    if "arma_utilizada" in df.columns and "tipo_crime" in df.columns:
        plt.figure(figsize=(14,6))
        sns.countplot(data=df, x="tipo_crime", hue="arma_utilizada", order=df["tipo_crime"].unique())
        plt.xticks(rotation=45)
        plt.title("Uso de armas por tipo de crime")
        plt.tight_layout()
        if save:
            plt.savefig("outputs/arma_por_tipo.png", bbox_inches="tight")
            print("📁 saved outputs/arma_por_tipo.png")

    # Placeholder para futuras visualizações de TF-IDF
    if "descricao_modus_operandi" in df.columns:
        print("ℹ️ Coluna 'descricao_modus_operandi' presente — pode ser usada para wordcloud ou feature importance.")

def heatmap_hotspots(df: pd.DataFrame, lat_col="latitude", lon_col="longitude", save_html=True, out_name="outputs/hotspots.html"):
    """Cria heatmap folium se latitude/longitude existirem."""
    if lat_col not in df.columns or lon_col not in df.columns:
        print("⚠️ Latitude/Longitude não disponíveis — heatmap pulado.")
        return None
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=12)
    pts = df[[lat_col, lon_col]].dropna().values.tolist()
    HeatMap(pts, radius=8).add_to(m)
    if save_html:
        m.save(out_name)
        print(f"📁 Hotspot salvo em {out_name}")
    return m
