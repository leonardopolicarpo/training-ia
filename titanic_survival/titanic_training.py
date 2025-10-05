import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- FASES 1, 2 e 3: Carregamento, Limpeza e Preparação ---
df_train = pd.read_csv('data/train.csv')
df_train.drop('Cabin', axis=1, inplace=True)
porto_mais_frequente = df_train['Embarked'].mode()[0]
df_train['Embarked'] = df_train['Embarked'].fillna(porto_mais_frequente)
idade_mediana = df_train['Age'].median()
df_train['Age'] = df_train['Age'].fillna(idade_mediana)
colunas_para_converter = ['Sex', 'Embarked']
df_train = pd.get_dummies(df_train, columns=colunas_para_converter, drop_first=True)
print("Dados carregados e pré-processados com sucesso.\n")


# --- FASE 4: TREINAMENTO E AVALIAÇÃO DO MODELO ---
print("--- 4. Iniciando treinamento do modelo ---")
y = df_train['Survived']
X = df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia do modelo no conjunto de teste: {acuracia * 100:.2f}%\n")


# --- FASE 5: ENTENDENDO O MODELO (FEATURE IMPORTANCE) ---
print("--- 5. Analisando a importância de cada feature ---")

# O modelo treinado tem um atributo .feature_importances_ que nos dá um score para cada feature.
importances = modelo.feature_importances_
# Para deixar isso legível, vamos criar um DataFrame do Pandas com os nomes das features e suas importâncias.
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Vamos ordenar o DataFrame para ver as features mais importantes no topo.
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Ranking de Importância das Features:")
print(feature_importance_df)

# E para ficar ainda melhor, vamos visualizar isso em um gráfico!
print("\nGerando gráfico de importância das features (salvo como 'feature_importance.png')...")
plt.figure(figsize=(10, 6)) # Define o tamanho da figura do gráfico
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importância de Cada Feature para Prever a Sobrevivência')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout() # Ajusta o layout para não cortar os nomes
plt.savefig('feature_importance.png') # Salva o gráfico como um arquivo de imagem
# plt.show() # Descomente esta linha se quiser que o gráfico apareça numa janela pop-up

print("Processo finalizado.")
