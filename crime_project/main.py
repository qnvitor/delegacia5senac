import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from scripts.preprocess import prepare_features, build_preprocessor
from scripts.utils import ensure_outputs

# Carregar dataset
df = pd.read_csv("data/dataset_ocorrencias_delegacia_5.csv")
df_modelo, X, y = prepare_features(df)

# Colunas
categoricas = ["bairro", "arma_utilizada", "sexo_suspeito"]
numericas = ["quantidade_vitimas","quantidade_suspeitos","idade_suspeito","mes","dia","ano","dia_semana","latitude","longitude"]
text_cols = ["descricao_modus_operandi"]

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Criar preprocessor completo e fitar
preprocessor = build_preprocessor(categoricas, numericas, text_cols)
preprocessor.fit(X_train)  # ⚠️ fit obrigatório
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_trans, y_train)

# Treinar RandomForest
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train_bal, y_train_bal)

# Avaliar
y_pred = rf_model.predict(X_test_trans)
print(classification_report(y_test, y_pred))

# Salvar modelos e preprocessor fitado
ensure_outputs()
joblib.dump(rf_model, "outputs/rf_model.pkl")
joblib.dump(preprocessor, "outputs/preprocessor.pkl")  # já fitado
print("✅ Modelos e preprocessor salvos com sucesso!")
