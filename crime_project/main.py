# ==========================================================
from scripts.utils import load_dataset, show_nulls
from scripts.visualization import plot_exploratory
from scripts.preprocess import prepare_features, handle_outliers, build_preprocessor
from scripts.modeling import train_models, compare_results, plot_confusion_matrix

from sklearn.model_selection import train_test_split

# 1️⃣ Carregamento
df = load_dataset("data/dataset_ocorrencias_delegacia_5.csv")
show_nulls(df)

# 2️⃣ Visualização inicial
plot_exploratory(df)

# 3️⃣ Pré-processamento
df_modelo, X, y = prepare_features(df)

numericas = ["quantidade_vitimas", "quantidade_suspeitos", "idade_suspeito", "mes", "dia", "ano", "dia_semana"]
categoricas = ["bairro", "descricao_modus_operandi", "arma_utilizada", "sexo_suspeito"]

X = handle_outliers(X, numericas)

# Split temporal simples (ajuste conforme dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
preprocessador = build_preprocessor(categoricas, numericas)

# 4️⃣ Modelagem
modelos = train_models(X_train, y_train, X_test, y_test, preprocessador)
df_resultados = compare_results(modelos, y_test)

# 5️⃣ Avaliação final (Random Forest)
rf_model, y_pred_rf = modelos["Random Forest"]
plot_confusion_matrix(rf_model, y_test, y_pred_rf)
