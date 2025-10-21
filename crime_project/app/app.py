# ==========================================================
# Streamlit App Atualizado
# ==========================================================
import streamlit as st
import pandas as pd
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =========================
# Carregamento de scripts
# =========================
from scripts.utils import ensure_outputs, load_dataset, check_columns
from scripts.preprocess import prepare_features, handle_outliers, build_preprocessor
from scripts.visualization import heatmap_hotspots, plot_exploratory
from scripts.clustering import embed_and_reduce, run_kmeans, run_hdbscan, describe_clusters, plot_clusters
from scripts.anomaly_detection import isolation_forest_detect, lof_detect
from scripts.modeling import train_classifiers

# ======================================
# Configuração inicial do Streamlit
# ======================================
st.set_page_config(
    page_title="Crime Data Analysis",
    layout="wide",
    page_icon="🔍"
)

st.title("📊 Crime Project Dashboard")
st.markdown("### Explore, descubra padrões e analise modelos de previsão de crimes.")

ensure_outputs()

# ======================================
# Upload ou dataset padrão
# ======================================
st.sidebar.header("⚙️ Configurações de Dados")
if "df" not in st.session_state:
    st.session_state["df"] = None

uploaded = st.sidebar.file_uploader("📁 Faça upload de um CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df
    st.sidebar.success("✅ Dataset carregado via upload.")
elif st.sidebar.button("📊 Usar dataset padrão"):
    sample_path = "data/dataset_ocorrencias_delegacia_5.csv"
    if os.path.exists(sample_path):
        df = load_dataset(sample_path)
        st.session_state["df"] = df
        st.sidebar.success("✅ Dataset padrão carregado.")
    else:
        st.sidebar.error("❌ Arquivo padrão não encontrado.")
        st.stop()

df = st.session_state["df"]
if df is None:
    st.sidebar.info("⬆️ Faça upload de um arquivo ou use o dataset padrão.")
    st.stop()

# ======================================
# Navegação principal
# ======================================
aba = st.sidebar.radio(
    "Escolha uma aba:",
    ["Exploração", "Clusters", "Anomalias", "Modelos Supervisionados", "Relatório Final"],
    format_func=lambda x: f"📘 {x}" if x == "Exploração" else
    f"🧩 {x}" if x == "Clusters" else
    f"⚠️ {x}" if x == "Anomalias" else
    f"🤖 {x}" if x == "Modelos Supervisionados" else
    f"📄 {x}"
)

# ======================================
# 1️⃣ Exploração de Dados
# ======================================
if aba == "Exploração":
    st.header("📊 Exploração de Dados")
    st.write("#### Amostra do Dataset")
    st.dataframe(df.head())
    st.write("#### Estatísticas gerais")
    st.dataframe(df.describe(include='all').T)

    st.write("#### Visualizações automáticas")
    plot_exploratory(df)
    st.pyplot(plt)

    if "latitude" in df.columns and "longitude" in df.columns:
        st.write("#### 🌎 Mapa de Hotspots")
        mapa = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)
        HeatMap(df[["latitude", "longitude"]].dropna()).add_to(mapa)
        st_folium(mapa, width=800, height=500)
    else:
        st.warning("Colunas de latitude/longitude não encontradas.")

# ======================================
# 2️⃣ Clusterização
# ======================================
elif aba == "Clusters":
    st.header("🧩 Análise de Clusters (KMeans)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("O dataset precisa ter ao menos duas colunas numéricas para clusterização.")
    else:
        n_clusters = st.slider("Número de clusters (KMeans)", 2, 10, 4)
        model, clusters, score = run_kmeans(df[numeric_cols], n_clusters)
        df["cluster"] = clusters
        st.success(f"Clusterização concluída! Silhouette Score = {score:.3f}")

        st.write("#### Visualização dos clusters (2D PCA aproximado)")
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], hue="cluster", data=df, palette="tab10")
        st.pyplot(plt)

        st.write("#### Tamanho de cada cluster")
        st.dataframe(df["cluster"].value_counts())

# ======================================
# 3️⃣ Anomalias
# ======================================
elif aba == "Anomalias":
    st.header("⚠️ Detecção de Anomalias (Isolation Forest)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("O dataset precisa ter ao menos duas colunas numéricas.")
    else:
        model, preds, scores = isolation_forest_detect(df[numeric_cols])
        df["anomalia"] = preds
        outliers = df[df["anomalia"] == True]

        st.write(f"🔍 Foram detectadas {len(outliers)} anomalias no total.")
        st.write("#### Distribuição das anomalias")
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df, hue="anomalia", palette="coolwarm")
        st.pyplot(plt)

        if "latitude" in df.columns and "longitude" in df.columns:
            st.write("#### 🌎 Mapa das Anomalias")
            mapa = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)
            HeatMap(outliers[["latitude", "longitude"]].dropna(), radius=10).add_to(mapa)
            st_folium(mapa, width=800, height=500)

# ======================================
# 4️⃣ Modelos supervisionados
# ======================================
elif aba == "Modelos Supervisionados":
    st.header("🤖 Previsão de Tipo de Crime")

    # Carregar modelo e preprocessor
    try:
        rf_model = joblib.load("outputs/rf_model.pkl")
        preprocessor = joblib.load("outputs/preprocessor.pkl")
    except FileNotFoundError as e:
        st.error(f"❌ Arquivo não encontrado: {e}")
        st.info("Execute primeiro o script main.py para treinar os modelos.")
        st.stop()

    st.write("Use o formulário abaixo para prever o tipo de crime:")

    # Inputs do usuário
    bairro = st.selectbox("Bairro", sorted(df["bairro"].dropna().unique()))
    arma = st.selectbox("Arma utilizada", sorted(df["arma_utilizada"].dropna().unique()))
    sexo_suspeito = st.selectbox("Sexo do suspeito", sorted(df["sexo_suspeito"].dropna().unique()))
    quantidade_vitimas = st.number_input("Quantidade de vítimas", min_value=0, max_value=10, value=1)
    quantidade_suspeitos = st.number_input("Quantidade de suspeitos", min_value=0, max_value=10, value=1)
    idade_suspeito = st.number_input("Idade do suspeito", min_value=0, max_value=100, value=30)
    descricao_modus_operandi = st.text_area("Descrição do Modus Operandi", "")

    if st.button("🔮 Prever Tipo de Crime"):
        if descricao_modus_operandi.strip() == "":
            descricao_modus_operandi = "sem descricao"

        # Criar DataFrame com os inputs do usuário
        X_new = pd.DataFrame([{
            "id_ocorrencia": "user_input_001",
            "data_ocorrencia": "2024-01-01",
            "bairro": bairro,
            "descricao_modus_operandi": descricao_modus_operandi,
            "arma_utilizada": arma,
            "quantidade_vitimas": quantidade_vitimas,
            "quantidade_suspeitos": quantidade_suspeitos,
            "sexo_suspeito": sexo_suspeito,
            "idade_suspeito": idade_suspeito,
            "orgao_responsavel": "Delegacia",
            "status_investigacao": "Em andamento",
            "latitude": 0.0,
            "longitude": 0.0
        }])

        try:
            # Preparar features (extrair componentes de data)
            _, X_processed, _ = prepare_features(X_new)
            
            # Aplicar preprocessor
            X_transformed = preprocessor.transform(X_processed)
            
            # Prever usando modelo
            y_pred = rf_model.predict(X_transformed)
            
            st.success(f"🔮 Tipo de crime previsto: {y_pred[0]}")
            
        except Exception as e:
            st.error(f"❌ Erro na previsão: {str(e)}")
            st.info("Verifique se os dados de entrada estão no formato correto.")

# ======================================
# 5️⃣ Relatório Final
# ======================================
elif aba == "Relatório Final":
    st.header("📄 Relatório Final do Projeto")
    st.markdown("""
        Este relatório resume as principais etapas do projeto:

        - **Exploração de Dados:** Análise descritiva, mapas de calor e padrões temporais.  
        - **Clusterização:** Agrupamento de ocorrências semelhantes por características numéricas.  
        - **Anomalias:** Identificação de registros fora do padrão.  
        - **Modelagem Supervisionada:** Previsão de tipo de crime usando Random Forest.  

        ---
    """)
    if st.button("📤 Exportar Relatório (HTML)"):
        html_path = "outputs/relatorio.html"
        df.describe().to_html(html_path)
        st.success(f"Relatório exportado para {html_path}")
