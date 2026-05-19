import math
import numpy as np
import matplotlib.pyplot as plt
#imports para o hdbscan
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal 


# Função para calcular a maior distância intra-cluster
def intra_cluster_dist(cluster_dataframe_list):

    # Inicializa a maior distância
    maxi = 0

    # Percorre cada cluster
    for df in cluster_dataframe_list:

        # Percorre todos os pontos do cluster
        for i in range(len(df)):

            # Calcula distância entre pares de pontos
            for ii in range(i + 1, len(df)):

                distance = math.dist(df[i], df[ii])

                # Atualiza a maior distância encontrada
                maxi = max(maxi, distance)

    return maxi


# Função para calcular a menor distância inter-cluster
def inter_cluster_dist(cluster_dataframe_list):

    # Inicializa com infinito
    mini = math.inf

    # Percorre os clusters
    for i in range(len(cluster_dataframe_list) - 1):

        for j in range(len(cluster_dataframe_list[i])):

            x = cluster_dataframe_list[i]

            # Percorre o próximo cluster
            for k in range(len(cluster_dataframe_list[i + 1])):

                y = cluster_dataframe_list[i + 1]

                # Distância entre pontos de clusters diferentes
                distance = math.dist(x[j], y[k])

                # Atualiza a menor distância encontrada
                mini = min(mini, distance)

    return mini


# Função para cálculo do índice de Dunn
def dunn_index(cluster_dataframe_list):

    # Distância mínima entre clusters
    numerator = inter_cluster_dist(cluster_dataframe_list)

    # Distância máxima dentro dos clusters
    denominator = intra_cluster_dist(cluster_dataframe_list)

    # Índice de Dunn
    return numerator / denominator


# Função para separar o DataFrame por clusters
def cluster_wise_df(dataframe, cluster_label_list):

    cluster_df_list = []

    # Percorre todos os clusters únicos
    for i in range(len(np.unique(cluster_label_list))):

        # Seleciona apenas os dados do cluster atual
        j = np.array(
            dataframe.loc[
                dataframe.Cluster == i
            ].drop(["Cluster"], axis=1)
        )

        cluster_df_list.append(j)

        j = None

    return cluster_df_list


# ============================
# Visualização do Silhouette Score
# ============================

from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans


# Função para gerar gráficos do Silhouette Score
def Silhouette_visual(data):

    # Criação da figura com múltiplos subplots
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(8, 10)
    )

    plt.grid(False)

    # Testa diferentes quantidades de clusters
    for i in [2, 3, 4, 5, 6, 7]:

        '''
        Cria instância do KMeans
        para diferentes números de clusters
        '''

        km = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=100,
            random_state=42
        )

        # Define posição do subplot
        q, mod = divmod(i, 2)

        '''
        Cria o visualizador do Silhouette Score
        e ajusta ao conjunto de dados
        '''

        visualizer = SilhouetteVisualizer(
            km,
            colors='yellowbrick',
            ax=ax[q - 1][mod]
        )

        visualizer.fit(data)

        # Configuração dos rótulos dos eixos
        visualizer.ax.set_xlabel(
            "Valores do coeficiente Silhouette"
        )

        visualizer.ax.set_ylabel(
            "Número de pontos"
        )

        # Salva a imagem
        plt.savefig(
            'Silhouette_Plot.png',
            dpi=700,
            bbox_inches='tight'
        )


# ============================
# Método do Cotovelo (Elbow Method)
# ============================

# Função para gerar o gráfico do método do cotovelo
def elbow_plot(data):

    distortions = []

    # Valores possíveis de k
    K = range(1, 10)

    # Calcula a distorção para cada valor de k
    for k in K:

        kmeanModel = KMeans(n_clusters=k)

        kmeanModel.fit(data)

        distortions.append(
            kmeanModel.inertia_
        )

    # Configuração da figura
    plt.figure(figsize=(6, 3))

    plt.grid(False)

    # Plotagem do gráfico
    plt.plot(K, distortions, 'bx-')

    plt.xlabel('k')

    plt.ylabel('Distorção')

    plt.title(
        'Método do Cotovelo mostrando o valor ótimo de k'
    )

    # Salva o gráfico
    plt.savefig(
        'Elbow_plot.png',
        dpi=700,
        bbox_inches='tight'
    )

    # Exibe o gráfico
    plt.show()

#função auxiliar para hdbscan
def calcular_anova(data, cluster_list, cont_list, ord_list, alpha=0.05):
    # Cria uma cópia do dataframe e adiciona a coluna de clusters
        df = data.copy()
        df['Cluster'] = cluster_list
    
    # IMPORTANTE: Ignorar o cluster de ruído (-1) gerado pelo DBSCAN/HDBSCAN.
    # O ruído não é um grupo clínico válido para ser comparado na ANOVA.
        df = df[df['Cluster'] != -1]
    
    # Identifica os clusters únicos válidos
        clusters_unicos = df['Cluster'].unique()
    
    # Se houver apenas 1 cluster (ou 0), não há como comparar
        if len(clusters_unicos) <= 1:
            return 0.0
        
        features_significativas = 0
        total_features = len(cont_list) + len(ord_list)
    
        if total_features == 0:
            return 0.0
        
    # 1. Teste para variáveis contínuas (ANOVA - f_oneway)
        for col in cont_list:
        # Separa os valores da coluna agrupados por cluster
            grupos = [df[df['Cluster'] == c][col].dropna() for c in clusters_unicos]
            try:
                stat, p_value = f_oneway(*grupos)
                if p_value < alpha:
                    features_significativas += 1
            except ValueError:
                pass # Ignora caso algum grupo esteja vazio
            
    # 2. Teste para variáveis ordinais (Kruskal-Wallis)
        for col in ord_list:
            grupos = [df[df['Cluster'] == c][col].dropna() for c in clusters_unicos]
            try:
                stat, p_value = kruskal(*grupos)
                if p_value < alpha:
                    features_significativas += 1
            except ValueError:
                pass
            
    # Calcula a porcentagem final
        porcentagem = (features_significativas / total_features) * 100
    
        return round(porcentagem, 2)