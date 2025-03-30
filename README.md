# MODEL BASICS LINEAR 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Criando dados corretamente
x = np.linspace(0, 1, 50).reshape(-1, 1)  # Formato correto para regressão
y = np.linspace(0, 1, 50)  # Valores de saída

# Criando e treinando o modelo
model = LinearRegression()
model.fit(x, y)

# Fazendo previsões
y_pred = model.predict(x)

# Criando DataFrame para o Seaborn
df = pd.DataFrame({"x": x.flatten(), "y": y, "y_pred": y_pred})

# Plotando regressão linear
sns.lmplot(x="x", y="y", data=df, height=7, line_kws={"color": "red"})
plt.scatter(df["x"], df["y_pred"], color="blue", label="Previsão")
plt.legend()
plt.show()
