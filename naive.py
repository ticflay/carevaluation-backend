# Importando as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Carregando a base de dados
data = pd.read_csv('../carevaluation/cardata.csv', header=None)

# Convertendo as características categóricas em numéricas
le_dict = {}
for i in range(data.shape[1]):
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])
    le_dict[i] = le

# Separando as características (X) e o rótulo (y)
X = data.drop([6], axis=1)
y = data[6]

# Dividindo a base de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Treinando o modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
print('Acurácia:', accuracy_score(y_test, y_pred))

#Ordem colunas: buying, maint, doors, persons, lug_boot, safety
# Características do novo carro
new_car = ['vhigh', 'vhigh', '5more', 'more', 'big', 'high']  # Exemplo de características

# Convertendo as características do novo carro para numéricas
new_car_encoded = [le_dict[i].transform([new_car[i]])[0] for i in range(len(new_car))]

# Fazendo a previsão
prediction = model.predict(np.array(new_car_encoded).reshape(1, -1))

# Mapeamento de volta para as classes originais
class_mapping = {i: class_name for i, class_name in enumerate(le_dict[6].classes_)}

# Imprimindo a previsão
print('A previsão para o novo carro é:', class_mapping[prediction[0]])

