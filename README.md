# Projeto de Comparação de Redes Neurais
Este projeto tem como objetivo comparar o desempenho de diferentes algoritmos de aprendizado de máquina na classificação de notícias. 

##### Projeto feito para obtenção de nota na matéria de Inteligência Artificial, do curso de Ciência da Computação, 5° Período, Goiânia-GO.

# Algoritmos utilizados:
- GaussianNB,
- MLP,
- DecisionTree,
- KNN,
- Regressão Linear,
- LDA,
- SVM,
- RandomForest,
- AdaBoost e
- QDA.

# Pré-requisitos
Antes de executar este projeto, é necessário rodar o código de criação do JSON de treinamento.

# Como executar
- Clone este repositório.
- Instale as dependências necessárias. Isso inclui pandas, requests, seaborn, numpy, sklearn, matplotlib e json.
- Execute o script Python fornecido.

# Descrição do Código
O código começa importando as bibliotecas necessárias e carregando os dados do arquivo JSON ‘dados_noticias.json’. Em seguida, ele cria um DataFrame com os dados e exibe as primeiras 5 linhas e as informações do DataFrame.

Os dados são então divididos em conjuntos de treinamento e teste, com 70% dos dados usados para treinamento e 30% para teste. Apenas a coluna “titulo” é usada para x_data.

Em seguida, o código define uma função mostrar_desempenho que treina o modelo fornecido, faz previsões e exibe um relatório de desempenho. Esta função é então chamada para cada um dos algoritmos de aprendizado de máquina mencionados acima.

Finalmente, o código gera três gráficos de barras para comparar o desempenho dos algoritmos em termos de precisão, tempo de treinamento e tempo de previsão.

# Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
