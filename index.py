# OBSERVAÇÃO: É necessário rodar o codigo de criamento do json de treinamento da aps antes de executar esse

# Obtendo Dataset
import json
import pandas as pd
import requests
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Carregando os dados do JSON
with open('dados_noticias.json', 'r', encoding='utf-8') as file:
    news_data = json.load(file)

# Criando DataFrame com os dados
df = pd.DataFrame(news_data)
print(df.columns)
print('\nMostrando os 5 primeiros registros:')
pd.options.display.max_columns = None
print(df.head(5))

print('\nMostrando as informações do DataFrame:')
df.info()

# Separando os dados de teste (30%) e de treino (70%)
from sklearn.model_selection import train_test_split

# Usando apenas a coluna "titulo" para x_data
x_data = df["title"]
y_data = df["classification"]

# Dividindo os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
x_train_vectorized = (vectorizer.fit_transform(x_train)).toarray()
x_test_vectorized = (vectorizer.transform(x_test)).toarray()

print("x train: ", x_train_vectorized.shape)
print("x test: ", x_test_vectorized.shape)

# Relatório de Desempenho
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from time import time
def mostrar_desempenho(x_train_vectorized, y_train, x_test_vectorized, y_test, model, name):
# Treinando modelo
	inicio = time()
	model.fit(x_train_vectorized, y_train)
	fim = time()
	tempo_treinamento = (fim - inicio)*1000
	# Prevendo dados
	inicio = time()
	y_predicted = model.predict(x_test_vectorized)
	fim = time()
	tempo_previsao = (fim - inicio)*1000
	print('Relatório Utilizando Algoritmo', name)
	print('\nMostrando Matriz de Confusão:')
	print(confusion_matrix(y_test, y_predicted))
	print('\nMostrando Relatório de Classificação:')
	print(metrics.classification_report(y_test, y_predicted))
	accuracy = accuracy_score(y_test, y_predicted)
	print('Accuracy:', accuracy)
	relatorio = metrics.classification_report(y_test, y_predicted, output_dict=True)
	print('Precision:', relatorio['macro avg']['precision'])
	print('Tempo de treinamento (ms):',tempo_treinamento)
	print('Tempo de previsão (ms):',tempo_previsao)
	return accuracy, tempo_treinamento, tempo_previsao

# Classificação com GaussianNB
from sklearn.naive_bayes import GaussianNB
acc_gnb, tt_gnb, tp_gnb = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, GaussianNB(), 'GaussianNB')

# Classificação com MLP
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
model_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000)
acc_mlp, tt_mlp, tp_mlp = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, model_mlp, 'MLP')

# Classificação com DecisionTree
from sklearn import tree
model_arvore = tree.DecisionTreeClassifier()
acc_dt, tt_dt, tp_dt = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, model_arvore, 'DecisionTree')

# Classificação com KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=17, p=12)
acc_knn, tt_knn, tp_knn = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, model_knn, 'KNN')

# Classificação com Regressão Logística
from sklearn.linear_model import LogisticRegression
acc_lr, tt_lr, tp_lr = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, LogisticRegression(), 'LR')

# Classificação com LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
acc_lda, tt_lda, tp_lda = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, LinearDiscriminantAnalysis(), 'LDA')

# Classificação com SVM
from sklearn.svm import SVC
acc_svm, tt_svm, tp_svm = mostrar_desempenho( x_train_vectorized, y_train, x_test_vectorized, y_test, SVC(), 'SVM')

# Classificação com RandomForest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
acc_rf, tt_rf, tp_rf = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, SVC(), 'RandomForest')

# Classificação com AdaBoost
from sklearn.ensemble import AdaBoostClassifier
acc_ada, tt_ada, tp_ada = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, AdaBoostClassifier(), 'AdaBoost')

# Classificação com QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model_qda = QuadraticDiscriminantAnalysis()
acc_qda, tt_qda, tp_qda = mostrar_desempenho(
	x_train_vectorized, y_train, x_test_vectorized, y_test, model_qda, 'QDA')

print("--------------------------------------------------------------------------------")

# Comparação de Desempenho em Accuracy
fig = plt.figure(figsize=(9, 4))
ax = fig.add_axes([0, 0, 1, 1])
algoritmos = ['GaussianNB', 'MLP', 'DecisionTree', 'KNN', 'Regressão Linear', 'LDA', 'SVM', 'RandomForest', 'AdaBoost', 'QDA']
accs = [acc_gnb, acc_mlp, acc_dt, acc_knn, acc_lr, acc_lda, acc_svm, acc_rf, acc_ada, acc_qda]
ax.bar(algoritmos, accs)
ax.set_title('Comparação de Desempenho em Accuracy')
plt.show()

print("--------------------------------------------------------------------------------")

# Comparação de Desempenho em Tempo de Treinamento
fig = plt.figure(figsize=(9, 4))
ax = fig.add_axes([0, 0, 1, 1])
tts = [tt_gnb, tt_mlp, tt_dt, tt_knn, tt_lr, tt_lda, tt_svm, tt_rf, tt_ada, tt_qda]
ax.bar(algoritmos, tts)
ax.set_title('Comparação de Tempo de Treinamento')
plt.show()

print("--------------------------------------------------------------------------------")

# Comparação de Desempenho em Tempo de Previsão
fig = plt.figure(figsize=(9, 4))
ax = fig.add_axes([0, 0, 1, 1])
tps = [tp_gnb, tp_mlp, tp_dt, tp_knn, tp_lr, tp_lda, tp_svm, tp_rf, tp_ada, tp_qda]
ax.bar(algoritmos, tps)
ax.set_title('Comparação de Desempenho em Tempo de Previsão')
plt.show()
