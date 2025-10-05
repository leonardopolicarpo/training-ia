# Importando apenas o essencial do scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

# --- Passo 1: Carregar os Dados ---
print("1. Carregando o dataset Iris...")
# O objeto 'iris' contém tanto os dados (as medidas) quanto os alvos (a espécie da flor)
iris = load_iris()
X = iris.data # As 4 features: comprimento/largura da pétala e sépala
y = iris.target # O alvo: 0, 1 ou 2, representando cada espécie
print("   - Dataset carregado com sucesso!\n")


# --- Passo 2: Separar os Dados ---
print("2. Separando os dados em conjuntos de treino e teste...")
# Isso é crucial para avaliar se o modelo realmente aprendeu ou só decorou
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("   - Dados separados: 70% para treino, 30% para teste.\n")


# --- Passo 3: Treinar o Modelo ---
# Usaremos Regressão Logística, um dos modelos mais simples e rápidos.
model = LogisticRegression(max_iter=200)

print("3. Iniciando o treinamento do modelo...")
print("   (Observe o uso de CPU e RAM agora!)")
start_time = time.time()

# A mágica acontece aqui, no método .fit()
# O modelo vai "olhar" para os dados de treino (X_train) e as respostas (y_train)
# para aprender a relação entre eles.
model.fit(X_train, y_train)

end_time = time.time()
print(f"   - Treinamento concluído em {end_time - start_time:.4f} segundos.\n")


# --- Passo 4: Avaliar o Modelo ---
print("4. Avaliando a performance do modelo...")
# Agora, usamos o modelo treinado para prever as respostas dos dados de teste (que ele nunca viu)
# e comparamos com as respostas reais (y_test) para calcular a acurácia.
accuracy = model.score(X_test, y_test)
print(f"   - Acurácia do modelo nos dados de teste: {accuracy * 100:.2f}%\n")

print(">>> Processo finalizado! <<<")
