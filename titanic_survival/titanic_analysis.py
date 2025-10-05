import pandas as pd

# Carregando o dataset de treino. O Pandas transforma o arquivo CSV
# em uma estrutura de dados chamada DataFrame.
print("Carregando o arquivo train.csv...")
df_train = pd.read_csv('data/train.csv')
print("Arquivo carregado com sucesso!\n")


# --- INÍCIO DA ANÁLISE EXPLORATÓRIA ---

# O comando .head() mostra 5 primeiras linhas para termos uma ideia da estrutura.
print("--- 1. Primeiras 5 linhas dos dados (head) ---")
print(df_train.head())
print("-" * 50 + "\n")


# Resumo técnico dos dados
# O comando .info() mostra:
# - Quantas linhas (Entries)
# - Cada uma das colunas (Column)
# - Quantos valores NÃO NULOS existem em cada coluna (Non-Null Count) 
# - O tipo de dado de cada coluna (Dtype) -> object = texto, int64 = inteiro, float64 = decimal
print("--- 2. Resumo técnico dos dados (info) ---")
df_train.info()
print("-" * 50 + "\n")


# Obtendo um resumo estatístico das colunas numéricas
# O comando .describe() calcula estatísticas básicas (média, desvio padrão, mínimo, máximo, etc.)
# para todas as colunas que contêm números.
print("--- 3. Resumo estatístico dos dados (describe) ---")
print(df_train.describe())
print("-" * 50 + "\n")
