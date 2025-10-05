import pandas as pd

# --- FASE 1: ANÁLISE EXPLORATÓRIA ---
print("Carregando o arquivo data/train.csv...")
df_train = pd.read_csv('data/train.csv')
print("Arquivo carregado com sucesso!\n")


# --- FASE 2: LIMPEZA DOS DADOS ---
print("--- 2. Iniciando limpeza dos dados ---")

# 2.1. Lidando com a coluna 'Cabin'
df_train.drop('Cabin', axis=1, inplace=True)

# 2.2. Lidando com a coluna 'Embarked'
porto_mais_frequente = df_train['Embarked'].mode()[0]
df_train['Embarked'] = df_train['Embarked'].fillna(porto_mais_frequente)

# 2.3. Lidando com a coluna 'Age'
idade_mediana = df_train['Age'].median()
df_train['Age'] = df_train['Age'].fillna(idade_mediana)

print("Limpeza de valores nulos concluída.\n")


# --- FASE 3: ENGENHARIA DE FEATURES (ENCODING) ---
print("--- 3. Convertendo colunas de texto para numéricas (One-Hot Encoding) ---")

# Selecionamos as colunas que queremos 'traduzir'
colunas_para_converter = ['Sex', 'Embarked']

# Usamos a função get_dummies do Pandas para fazer a mágica.
# drop_first=True é uma boa prática para evitar redundância (se não é male, tem que ser female).
df_train = pd.get_dummies(df_train, columns=colunas_para_converter, drop_first=True)

print("Colunas convertidas com sucesso.\n")


# --- VERIFICAÇÃO FINAL ---
print("--- 4. Resumo técnico FINAL ---")
df_train.info()
print("-" * 50 + "\n")

print("--- 5. Primeiras 5 linhas dos dados FINAIS ---")
# Observe as novas colunas no final do DataFrame!
print(df_train.head())
print("-" * 50 + "\n")

