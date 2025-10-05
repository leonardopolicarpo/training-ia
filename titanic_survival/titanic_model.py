import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("--- INICIANDO PROCESSO COMPLETO DE ANÁLISE E TREINAMENTO ---\n")

# --- FASE 1: CARREGAMENTO DOS DADOS ---
print("FASE 1: Carregando dados de 'data/train.csv'...")
df_train = pd.read_csv('data/train.csv')
print("Dados carregados com sucesso.\n")


# --- FASE 2: LIMPEZA DOS DADOS (DATA CLEANING) ---
print("FASE 2: Iniciando limpeza de valores nulos...")
# 2.1. Lidando com a coluna 'Cabin' (muitos dados faltantes)
df_train.drop('Cabin', axis=1, inplace=True)

# 2.2. Lidando com a coluna 'Embarked' (poucos dados faltantes)
porto_mais_frequente = df_train['Embarked'].mode()[0]
df_train['Embarked'] = df_train['Embarked'].fillna(porto_mais_frequente)

# 2.3. Lidando com a coluna 'Age' (dados moderadamente faltantes)
idade_mediana = df_train['Age'].median()
df_train['Age'] = df_train['Age'].fillna(idade_mediana)
print("Limpeza de valores nulos concluída.\n")


# --- FASE 3: ENGENHARIA DE FEATURES (ENCODING) ---
print("FASE 3: Convertendo colunas de texto para numéricas...")
colunas_para_converter = ['Sex', 'Embarked']
df_train = pd.get_dummies(df_train, columns=colunas_para_converter, drop_first=True)
print("Conversão concluída.\n")


# --- FASE 4: TREINAMENTO E AVALIAÇÃO DO MODELO ---
print("FASE 4: Iniciando treinamento do modelo...")

# 4.1. Separando Features (X) e Target (y)
y = df_train['Survived']
# Removemos a coluna target e as colunas que não ajudam na predição
X = df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)

# 4.2. Separando Dados de Treino e Teste para avaliação interna
# random_state=42 garante que a divisão seja sempre a mesma (reprodutibilidade)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dados divididos em {len(X_train)} para treino e {len(X_test)} para teste.")

# 4.3. Treinando o Modelo RandomForestClassifier
# random_state=42 também garante a reprodutibilidade do modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
print("Modelo treinado com sucesso.")

# 4.4. Avaliando a Performance
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia do modelo no conjunto de teste: {acuracia * 100:.2f}%")

# 4.5. Salvando o Modelo Treinado
print("Salvando o modelo treinado em 'titanic_model.pkl'...")
joblib.dump(modelo, 'titanic_model.pkl')
print("Modelo salvo com sucesso!\n")


# --- FASE 5: ANÁLISE DE IMPORTÂNCIA DAS FEATURES ---
print("FASE 5: Analisando a importância de cada feature...")
importances = modelo.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Ranking de Importância das Features:")
print(feature_importance_df)

# Visualização do Ranking
print("\nGerando gráfico de importância (salvo como 'feature_importance.png')...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importância de Cada Feature para Prever a Sobrevivência')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Gráfico salvo com sucesso.")

print("\n--- PROCESSO FINALIZADO ---")
