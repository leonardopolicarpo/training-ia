import pandas as pd

# --- FASE 1: ANÁLISE EXPLORATÓRIA ---

print("Carregando o arquivo data/train.csv...")
df_train = pd.read_csv('data/train.csv')
print("Arquivo carregado com sucesso!\n")


# --- FASE 2: LIMPEZA E PREPARAÇÃO DOS DADOS ---

print("--- 2. Iniciando limpeza dos dados ---")

# 2.1. Lidando com a coluna 'Cabin'
print("Passo 2.1: Removendo a coluna 'Cabin'...")
df_train.drop('Cabin', axis=1, inplace=True)
print("Coluna 'Cabin' removida.\n")

# 2.2. Lidando com a coluna 'Embarked'
print("Passo 2.2: Preenchendo valores faltantes em 'Embarked'...")
porto_mais_frequente = df_train['Embarked'].mode()[0]
print(f"O porto mais frequente é: {porto_mais_frequente}")
# CORREÇÃO: Usando o método recomendado para evitar o FutureWarning
df_train['Embarked'] = df_train['Embarked'].fillna(porto_mais_frequente)
print("Valores faltantes em 'Embarked' preenchidos.\n")

# 2.3. Lidando com a coluna 'Age'
print("Passo 2.3: Preenchendo valores faltantes em 'Age' com a mediana...")
# Primeiro, calculamos a mediana das idades existentes.
idade_mediana = df_train['Age'].median()
print(f"A idade mediana é: {idade_mediana:.2f} anos")
# Agora, preenchemos os valores nulos com essa mediana.
df_train['Age'] = df_train['Age'].fillna(idade_mediana)
print("Valores faltantes em 'Age' preenchidos.\n")


# --- VERIFICAÇÃO PÓS-LIMPEZA ---
print("--- 3. Resumo técnico FINAL ---")
# Verificando o .info() novamente. Todas as colunas agora devem ter 891 non-null.
df_train.info()
print("-" * 50 + "\n")
