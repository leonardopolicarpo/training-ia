import pandas as pd
import joblib

print("--- INICIANDO PROCESSO DE PREDIÇÃO ---")

# --- ETAPA 1: CARREGAR O MODELO E OS NOVOS DADOS ---
print("Carregando o modelo treinado de 'titanic_model.pkl'...")
try:
    modelo = joblib.load('titanic_model.pkl')
except FileNotFoundError:
    print("\nERRO: O arquivo 'titanic_model.pkl' não foi encontrado.")
    print("Por favor, execute o script 'titanic_analysis.py' primeiro para treinar e salvar o modelo.")
    exit()

print("Carregando os dados de teste de 'data/test.csv'...")
df_test = pd.read_csv('data/test.csv')
# Guardamos os IDs dos passageiros, pois precisaremos deles no arquivo final
passenger_ids = df_test['PassengerId']
print("Modelo e dados de teste carregados com sucesso.\n")


# --- ETAPA 2: APLICAR O MESMO PRÉ-PROCESSAMENTO DOS DADOS DE TREINO ---
# Esta é a etapa mais crucial. O modelo só entende dados no mesmo formato em que foi treinado.
print("Aplicando o mesmo pré-processamento dos dados de treino...")

# 2.1. Lidando com valores nulos
# Usamos a mediana do próprio conjunto de teste. Em um cenário de produção mais avançado,
# salvaríamos a mediana do conjunto de treino para usar aqui, garantindo consistência total.
idade_mediana_test = df_test['Age'].median()
df_test['Age'] = df_test['Age'].fillna(idade_mediana_test)

# O arquivo de teste tem um valor de 'Fare' (tarifa) faltando.
fare_mediana_test = df_test['Fare'].median()
df_test['Fare'] = df_test['Fare'].fillna(fare_mediana_test)

# 2.2. Encoding das colunas categóricas (One-Hot Encoding)
colunas_para_converter = ['Sex', 'Embarked']
df_test = pd.get_dummies(df_test, columns=colunas_para_converter, drop_first=True)

# 2.3. Garantindo que as colunas sejam as mesmas do treino
# O modelo foi treinado esperando um conjunto específico de colunas. Vamos garantir que nosso
# dataframe de teste tenha exatamente essas colunas, na mesma ordem.
features_do_modelo = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
df_test_final = df_test[features_do_modelo]

print("Pré-processamento concluído.\n")


# --- ETAPA 3: FAZER AS PREVISÕES ---
print("Fazendo previsões nos dados de teste...")
previsoes_finais = modelo.predict(df_test_final)
print("Previsões geradas com sucesso.\n")


# --- ETAPA 4: CRIAR O ARQUIVO DE SUBMISSÃO ---
print("Criando o arquivo de submissão 'submission.csv'...")
# Criamos um novo DataFrame com o ID do passageiro e a previsão de sobrevivência.
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': previsoes_finais
})

# Salvamos o DataFrame em um arquivo .csv, sem o índice do pandas.
submission_df.to_csv('submission.csv', index=False)

print("--- PROCESSO FINALIZADO! ---")
print("O arquivo 'submission.csv' foi criado na pasta do projeto.")
print("Ele contém as previsões de sobrevivência para os passageiros do conjunto de teste.")
