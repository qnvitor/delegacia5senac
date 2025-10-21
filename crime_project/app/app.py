import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main
from scripts.utils import load_dataset
from scripts.preprocess import prepare_features
from scripts.visualization import plot_exploratory
from scripts.clustering import run_kmeans
from scripts.anomaly_detection import isolation_forest_detect


st.set_page_config(page_title="Delegacia Senac", layout="wide", page_icon="🔍")
st.title("📊 Delegacia Senac Dashboard")
st.markdown("### Explore, descubra padrões e analise modelos de previsão de crimes.")

with st.spinner("Preparando os dados..."):

    #Verifica se os arquivos gerados pelo main.py já existem
    preprocessed_path = "outputs/preprocessed_data.pkl"
    model_path = "outputs/rf_model.pkl"

    if not os.path.exists(preprocessed_path) or not os.path.exists(model_path):
        # Só roda o main se os arquivos não existirem
        main.run()
        st.success("✅ Pré-processamento e treinamento concluídos!")
    else:
        st.info("Dados e modelos já preparados. Pulando execução do main.py.")


@st.cache_data(show_spinner=False)
def load_default_dataset():
    return load_dataset("data/dataset_ocorrencias_delegacia_5.csv")

@st.cache_data(show_spinner=False)
def load_uploaded_csv(upload):
    return pd.read_csv(upload)

@st.cache_resource(show_spinner=False)
def load_model_and_preprocessor():
    rf_model = joblib.load("outputs/rf_model.pkl")
    preprocessor = joblib.load("outputs/preprocessor.pkl")
    return rf_model, preprocessor


st.sidebar.header("⚙️ Configurações de Dados")
if "df" not in st.session_state:
    st.session_state["df"] = None

uploaded = st.sidebar.file_uploader("📁 Faça upload de um CSV", type=["csv"])

if uploaded:
    with st.spinner("📦 Lendo arquivo CSV..."):
        st.session_state["df"] = load_uploaded_csv(uploaded)
        st.sidebar.success("✅ Dataset carregado via upload.")
elif st.sidebar.button("📊 Usar dataset padrão"):
    with st.spinner("📂 Carregando dataset padrão..."):
        if os.path.exists("data/dataset_ocorrencias_delegacia_5.csv"):
            st.session_state["df"] = load_default_dataset()
            st.sidebar.success("✅ Dataset padrão carregado.")
        else:
            st.sidebar.error("❌ Arquivo padrão não encontrado.")
            st.stop()

df = st.session_state["df"]
if df is None:
    st.markdown("""
    <div style='text-align:center;padding:50px'>
        <div class="loader"></div>
        <p>Carregue um dataset para começar.</p>
    </div>
    <style>
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin:auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    st.stop()


aba = st.sidebar.radio(
    "Escolha uma aba:",
    ["Modelos Supervisionados", "Exploração", "Clusters", "Anomalias", "Relatório Final"],
    format_func=lambda x: {
        "Modelos Supervisionados": "🚀 Previsão",
        "Exploração": "📘 Exploração",
        "Clusters": "🧩 Clusters",
        "Anomalias": "⚠️ Anomalias",
        "Relatório Final": "📄 Relatório Final",
    }[x],
)

if aba == "Modelos Supervisionados":
    st.header("🚀 Previsão de Tipo de Crime")
    try:
        with st.spinner("Carregando modelo..."):
            rf_model, preprocessor = load_model_and_preprocessor()
    except FileNotFoundError as e:
        st.error(f"❌ Arquivo não encontrado: {e}")
        st.info("Execute primeiro o script main.py para treinar os modelos.")
        st.stop()

    st.write("Use o formulário abaixo para prever o tipo de crime:")
    bairro = st.selectbox("Bairro", sorted(df["bairro"].dropna().unique()))
    arma = st.selectbox("Arma utilizada", sorted(df["arma_utilizada"].dropna().unique()))
    sexo_suspeito = st.selectbox("Sexo do suspeito", sorted(df["sexo_suspeito"].dropna().unique()))
    quantidade_vitimas = st.number_input("Quantidade de vítimas", min_value=0, max_value=10, value=1)
    quantidade_suspeitos = st.number_input("Quantidade de suspeitos", min_value=0, max_value=10, value=1)
    idade_suspeito = st.number_input("Idade do suspeito", min_value=0, max_value=100, value=30)
    descricao_modus_operandi = st.text_area("Descrição do Modus Operandi", "")

    if st.button("🔮 Prever Tipo de Crime"):
        with st.spinner("Realizando previsão..."):
            if descricao_modus_operandi.strip() == "":
                descricao_modus_operandi = "sem descricao"

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
                _, X_processed, _ = prepare_features(X_new)
                X_transformed = preprocessor.transform(X_processed)
                y_pred = rf_model.predict(X_transformed)
                st.success(f"🔮 Tipo de crime previsto: **{y_pred[0]}**")
            except Exception as e:
                st.error(f"❌ Erro na previsão: {str(e)}")
                st.info("Verifique se os dados de entrada estão no formato correto.")

elif aba == "Exploração":
    st.header("📊 Exploração de Dados")

    with st.spinner("Gerando estatísticas e gráficos..."):
        st.write("#### Amostra do Dataset")
        st.dataframe(df.head())

        st.write("#### Estatísticas gerais")
        st.dataframe(df.describe(include='all').T)
        st.write("#### Visualizações automáticas")
        plot_exploratory(df)

        if "latitude" in df.columns and "longitude" in df.columns:
            st.write("#### 🌎 Mapa de Hotspots")
            mapa = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)
            HeatMap(df[["latitude", "longitude"]].dropna()).add_to(mapa)
            st_folium(mapa, width=800, height=500)
        else:
            st.warning("Colunas de latitude/longitude não encontradas.")
            st.pyplot(plt.gcf())
            plt.clf()

elif aba == "Clusters":
    st.header("🧩 Análise de Clusters (KMeans)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("O dataset precisa ter ao menos duas colunas numéricas para clusterização.")
    else:
        n_clusters = st.slider("Número de clusters (KMeans)", 2, 10, 4)
        with st.spinner("Executando KMeans..."):
            model, clusters, score = run_kmeans(df[numeric_cols], n_clusters)
            df["cluster"] = clusters
            st.success(f"Clusterização concluída! Silhouette Score = {score:.3f}")

        st.write("#### Visualização dos clusters (2D PCA aproximado)")
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], hue="cluster", data=df, palette="tab10")
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("#### Tamanho de cada cluster")
        st.dataframe(df["cluster"].value_counts())

elif aba == "Anomalias":
    st.header("⚠️ Detecção de Anomalias (Isolation Forest)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("O dataset precisa ter ao menos duas colunas numéricas.")
    else:
        with st.spinner("Detectando anomalias..."):
            model, preds, scores = isolation_forest_detect(df[numeric_cols])
            df["anomalia"] = preds
            outliers = df[df["anomalia"] == True]
            st.success(f"🔍 Foram detectadas {len(outliers)} anomalias.")

        st.write("#### Distribuição das anomalias")
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df, hue="anomalia", palette="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        if {"latitude", "longitude"}.issubset(df.columns):
            st.write("#### 🌎 Mapa das Anomalias")
            mapa = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)
            HeatMap(outliers[["latitude", "longitude"]].dropna(), radius=10).add_to(mapa)
            st_folium(mapa, width=800, height=500)

elif aba == "Relatório Final":
    st.header("📄 Relatório Final do Projeto")
    st.markdown("""
    Este relatório resume as principais etapas do projeto:
    - **Exploração de Dados:** Análise descritiva, mapas de calor e padrões temporais.
    - **Clusterização:** Agrupamento de ocorrências semelhantes.
    - **Anomalias:** Identificação de registros fora do padrão.
    - **Modelagem Supervisionada:** Previsão de tipo de crime usando Random Forest.
    ---
    """)

    if st.button("📤 Exportar Relatório (HTML)"):
        with st.spinner("Gerando relatório HTML..."):
            html_path = "outputs/relatorio.html"
            df.describe().to_html(html_path)
            time.sleep(1)
            st.success(f"✅ Relatório exportado para `{html_path}`")
            with open(html_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "⬇️ Baixar Relatório HTML",
                    data=f.read(),
                    file_name="relatorio.html",
                     mime="text/html"
                 )
