#!/usr/bin/env python
# coding: utf-8

# Importação das bibliotecas necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
from fdc.fdc import feature_clustering, value
import seaborn as sns


# Função responsável por criar e compilar a rede neural
def neural_network(n_features, hidden_dim1, hidden_dim2,
                   out_emb_size, act1, act2, loss):

    # Define sementes aleatórias para reprodutibilidade
    np.random.seed(42)
    tf.random.set_seed(42)

    # Construção da arquitetura da rede neural
    model = keras.Sequential([
        keras.layers.Dense(hidden_dim1,
                           input_dim=n_features,
                           activation=act1),

        keras.layers.Dense(hidden_dim2,
                           activation=act2),

        keras.layers.Dense(out_emb_size)
    ])

    # Compilação do modelo
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=['mse']
    )

    return model


# Função para decodificar os rótulos codificados dos clusters
def label_decoder(label_dataframe):

    # Converte DataFrame para array NumPy
    label_array = np.array(label_dataframe)

    decoded_labels = []

    # Seleciona o índice de maior valor em cada linha
    for i in label_array:
        max_val = np.argmax(i)
        decoded_labels.append(max_val)

    return decoded_labels


# Função para visualização dos clusters
def plotting(train_data_high_dim,
             predicted_data_high_dim,
             decoded_train_label,
             predicted_train_label,
             count):

    # Concatena embeddings reais e previstos
    concatenated_5dim = pd.concat([
        train_data_high_dim,
        predicted_data_high_dim
    ])

    # Redução de dimensionalidade usando UMAP
    two_dim_viz = feature_clustering(
        15,
        0.1,
        'euclidean',
        concatenated_5dim,
        0
    )

    # Junta os rótulos reais e previstos
    concatenated_cluster_labels = np.concatenate([
        np.array(decoded_train_label),
        np.array(predicted_train_label)
        + len(np.unique(predicted_train_label))
    ])

    # Adiciona os rótulos ao DataFrame
    two_dim_viz['Cluster'] = concatenated_cluster_labels

    # Cores escuras para clusters de treino
    darkerhues = [
        'lightcoral',
        'cornflowerblue',
        'orange',
        'mediumorchid',
        'lightseagreen',
        'olive',
        'chocolate',
        'steelblue'
    ]

    colors_set2 = []

    for i in range(len(np.unique(predicted_train_label))):
        colors_set2.append(darkerhues[i])

    # Adiciona cores claras correspondentes aos clusters previstos
    colors_set2 = colors_set2 + [
        "lightpink",
        'skyblue',
        'wheat',
        "plum",
        "paleturquoise",
        "lightgreen",
        'burlywood',
        'lightsteelblue'
    ]

    print(
        'Visualização do FDC para o conjunto de treino '
        '(tons escuros) e clusters previstos pela rede neural '
        '(tons claros) - Fold ' + str(count + 1)
    )

    # Plotagem dos clusters
    sns.lmplot(
        x="UMAP_0",
        y="UMAP_1",
        data=two_dim_viz,
        fit_reg=False,
        legend=False,
        hue='Cluster',
        scatter_kws={"s": 3},
        palette=sns.set_palette(
            sns.color_palette(colors_set2)
        )
    )

    plt.show()


# Função para cálculo de métricas por cluster
def cluster_wise_F1score(ref_list, pred_list):

    # Divisão segura para evitar divisão por zero
    def safeDiv(a, b):
        if b != 0:
            return a / b
        return 0.0

    F1_score_list = []
    Geometric_mean_list = []
    cluster_score_list = []

    true_positive_total = 0

    # Percorre cada cluster
    for i in np.unique(ref_list):

        # Obtém índices do cluster atual
        indices = [
            j for j, val in enumerate(ref_list)
            if val == i
        ]

        true_positive = 0

        # Conta predições corretas
        for index in indices:
            if i == pred_list[index]:
                true_positive += 1

        true_positive_total += true_positive

        # Calcula precisão
        precision = safeDiv(
            true_positive,
            pred_list.count(i)
        )

        # Calcula recall
        recall = safeDiv(
            true_positive,
            len(indices)
        )

        # Calcula F1-Score
        F1_score = safeDiv(
            2.0 * precision * recall,
            precision + recall
        )

        # Média geométrica
        GM = np.sqrt(precision * recall)

        # Percentual de acertos
        cluster_score = recall * 100.0

        print(
            "F1-Score do cluster "
            + str(i)
            + " é {}".format(F1_score)
        )

        print(
            "Média geométrica do cluster "
            + str(i)
            + " é {}".format(GM)
        )

        print(
            "Pontos corretamente previstos no cluster "
            + str(i)
            + " é {}%".format(cluster_score)
        )

        print("\n")

        # Armazena métricas ponderadas
        F1_score_list.append(
            (ref_list.count(i) / len(ref_list))
            * F1_score
        )

        Geometric_mean_list.append(
            (ref_list.count(i) / len(ref_list))
            * GM
        )

        cluster_score_list.append(
            (ref_list.count(i) / len(ref_list))
            * cluster_score
        )

    print(
        "Média ponderada do F1-Score de todos os clusters é {}".format(
            np.sum(F1_score_list)
        )
    )

    print(
        "Média ponderada da média geométrica de todos os clusters é {}".format(
            np.sum(Geometric_mean_list)
        )
    )

    print(
        "Média ponderada dos pontos corretamente previstos "
        "em todos os clusters é {}%".format(
            np.sum(cluster_score_list)
        )
    )


# Classe responsável pelos modelos de Machine Learning
class Neural_Network_model:

    # Inicialização da classe
    def __init__(self, X_train, X_test, y0, y1):

        self.X_train = X_train
        self.X_test = X_test
        self.y0 = y0
        self.y1 = y1

    # Modelo de regressão usando rede neural
    def NN_1(self,
             input_layer=None,
             hidden_layer_1=None,
             hidden_layer_2=None,
             output_layer=None,
             activation_1=None,
             activation_2=None,
             loss=None):

        # Define valores padrão caso não sejam informados
        input_layer = value(
            input_layer,
            len(self.X_train[0])
        )

        hidden_layer_1 = value(
            hidden_layer_1,
            int(0.6 * len(self.X_train[0]))
        )

        hidden_layer_2 = value(
            hidden_layer_2,
            int(0.36 * len(self.X_train[0]))
        )

        output_layer = value(
            output_layer,
            len(self.y0[0])
        )

        activation_1 = value(
            activation_1,
            "relu"
        )

        activation_2 = value(
            activation_2,
            "sigmoid"
        )

        loss = value(
            loss,
            "mse"
        )

        # Criação da rede neural
        model_1 = neural_network(
            input_layer,
            hidden_layer_1,
            hidden_layer_2,
            output_layer,
            activation_1,
            activation_2,
            loss
        )

        # Treinamento do modelo
        history = model_1.fit(
            self.X_train,
            self.y0,
            epochs=30,
            batch_size=8
        )

        print('\n')
        print(
            'Histórico de treinamento ao longo das épocas'
        )

        # Plotagem do erro MSE
        plt.plot(history.history['mse'], 'r')
        plt.ylabel('mse')
        plt.xlabel('época')
        plt.show()

        # Predição do embedding de alta dimensão
        predicted_high_dim = pd.DataFrame(
            model_1.predict(self.X_test),
            columns=[
                'c' + str(i + 1)
                for i in range(np.shape(self.y0)[1])
            ]
        )

        # Redução para baixa dimensão
        predicted_low_dim = feature_clustering(
            30,
            0.01,
            "euclidean",
            predicted_high_dim,
            False
        )

        return predicted_high_dim, predicted_low_dim

    # Modelo de classificação usando rede neural
    def NN_2(self,
             input_layer=None,
             hidden_layer_1=None,
             hidden_layer_2=None,
             output_layer=None,
             activation_1=None,
             activation_2=None,
             loss=None):

        input_layer = value(
            input_layer,
            len(self.X_train[0])
        )

        hidden_layer_1 = value(
            hidden_layer_1,
            int(0.6 * len(self.X_train[0]))
        )

        hidden_layer_2 = value(
            hidden_layer_2,
            int(0.36 * len(self.X_train[0]))
        )

        output_layer = value(
            output_layer,
            len(self.y1[0])
        )

        activation_1 = value(
            activation_1,
            "relu"
        )

        activation_2 = value(
            activation_2,
            "sigmoid"
        )

        loss = value(
            loss,
            "mse"
        )

        # Cria o modelo
        model_2 = neural_network(
            input_layer,
            hidden_layer_1,
            hidden_layer_2,
            output_layer,
            activation_1,
            activation_2,
            loss
        )

        # Treina o modelo
        history = model_2.fit(
            self.X_train,
            self.y1,
            epochs=30,
            batch_size=8
        )

        print('\n')
        print(
            'Histórico de treinamento ao longo das épocas'
        )

        plt.plot(history.history['mse'], 'r')
        plt.ylabel('mse')
        plt.xlabel('época')
        plt.show()

        # Predição dos clusters
        predicted_clusters = pd.DataFrame(
            model_2.predict(self.X_test)
        )

        # Decodificação dos clusters previstos
        decoded_predicted_clusters = label_decoder(
            predicted_clusters
        )

        return decoded_predicted_clusters

    # Regressão usando Gradient Boosting
    def GB_reg(self):

        regr = MultiOutputRegressor(
            GradientBoostingRegressor(
                random_state=42
            )
        ).fit(self.X_train, self.y0)

        # Predição do embedding
        reg_predicted_high_dim = pd.DataFrame(
            regr.predict(self.X_test),
            columns=[
                'c' + str(i + 1)
                for i in range(np.shape(self.y0)[1])
            ]
        )

        # Redução de dimensionalidade
        predicted_low_dim = feature_clustering(
            30,
            0.01,
            "euclidean",
            reg_predicted_high_dim,
            False
        )

        return reg_predicted_high_dim, predicted_low_dim

    # Classificação usando Gradient Boosting
    def GB_clf(self):

        clf = GradientBoostingClassifier(
            n_estimators=20,
            learning_rate=0.5,
            max_features=2,
            max_depth=2,
            random_state=42
        )

        # Treinamento do classificador
        clf.fit(self.X_train, self.y1)

        # Predição dos clusters
        clf_predicted_clusters = clf.predict(self.X_test)

        return clf_predicted_clusters