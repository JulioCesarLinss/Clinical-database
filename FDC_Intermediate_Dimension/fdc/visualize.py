#!/usr/bin/env python
# coding: utf-8

# Importação das bibliotecas necessárias

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Função para plotar clusters em um gráfico bidimensional
def plotCluster(
    data,
    clusterName="cluster",
    xName='FDC_1',
    yName='FDC_2',
    stroke=20
):

    # Conjunto de cores utilizadas nos clusters
    colors_set = [
        'lightcoral',
        'cornflowerblue',
        'orange',
        'mediumorchid',
        'lightseagreen',
        'olive',
        'chocolate',
        'steelblue',
        'paleturquoise',
        'lightgreen',
        'burlywood',
        'lightsteelblue'
    ]

    # Define a paleta de cores personalizada
    customPalette_set = sns.set_palette(
        sns.color_palette(colors_set)
    )

    # Remove a grade do fundo do gráfico
    sns.set_style(
        "whitegrid",
        {'axes.grid': False}
    )

    # Criação do gráfico de dispersão
    sns.lmplot(
        x=data.columns[0],
        y=data.columns[1],
        data=data,
        fit_reg=False,
        legend=True,
        hue=clusterName,
        scatter_kws={"s": stroke},
        palette=customPalette_set
    )

    # Salva a imagem do gráfico
    plt.savefig(
        'K_means_cluster_lung_cancer.png',
        dpi=700,
        bbox_inches='tight'
    )

    # Exibe o gráfico
    plt.show()


# Função para visualização do mapeamento UMAP
def plotMapping(
    data,
    xName="UMAP_0",
    yName="UMAP_1"
):

    # Paleta de cores utilizada no mapeamento
    colors_set1 = [
        "lightcoral",
        "lightseagreen",
        "mediumorchid",
        "orange",
        "burlywood",
        "cornflowerblue",
        "plum",
        "yellowgreen"
    ]

    # Define a paleta personalizada
    customPalette_set1 = sns.set_palette(
        sns.color_palette(colors_set1)
    )

    # Configuração do estilo do gráfico
    sns.set_style(
        "whitegrid",
        {'axes.grid': False}
    )

    # Criação do gráfico de dispersão UMAP
    sns.lmplot(
        x=xName,
        y=yName,
        data=data,
        fit_reg=False,
        legend=False,
        scatter_kws={"s": 3},
        palette=customPalette_set1
    )

    # Salva o gráfico
    plt.savefig(
        'fdc_cluster.png',
        dpi=700,
        bbox_inches='tight'
    )

    # Exibe o gráfico
    plt.show()


# Função para visualização detalhada das features
def vizx(
    feature_list,
    cluster_df_list,
    main_data,
    umap_data,
    cont_features,
    rev_dict,
    xName="FDC_1",
    yName="FDC_2"
):

    # Limite de categorias para visualização
    vizlimit = 15

    # Tamanho padrão das figuras
    plt.rcParams["figure.figsize"] = (12, 6)

    # Paleta de cores
    col = sns.color_palette("Set2")

    # Configuração da grade de subplots
    rows = 3
    columns = 3

    # Percorre cada feature
    for feature in feature_list:

        print('Nome da feature:', feature.upper())
        print('\n')

        # Verifica se a feature possui poucas categorias
        if len(main_data[feature].value_counts()) <= vizlimit:

            # Exibe distribuição de frequência para cada cluster
            for cluster_counter, cluster in enumerate(cluster_df_list):

                print(
                    'Distribuição de frequência do Cluster '
                    + str(cluster_counter + 1)
                )

                # Verifica se existe dicionário reverso
                if feature in list(rev_dict.keys()):

                    feat_keys = rev_dict[feature]

                    # Inverte chave e valor do dicionário
                    r = dict(
                        zip(
                            feat_keys.values(),
                            feat_keys.keys()
                        )
                    )

                    print(
                        cluster.replace({feature: r})[
                            feature
                        ].value_counts()
                    )

                else:
                    print(
                        cluster[feature].value_counts()
                    )

                print('\n')

            print('\n')
            print('\n')

            # Armazena os dados para o gráfico de barras
            cluster_bar = []

            for cluster in cluster_df_list:

                if feature in list(rev_dict.keys()):

                    y = np.array(
                        cluster.replace({feature: r})[
                            feature
                        ].value_counts()
                    )

                    x = np.array(
                        cluster.replace({feature: r})[
                            feature
                        ].value_counts().index
                    )

                    cluster_bar.append([x, y])

                else:

                    y = np.array(
                        cluster[feature]
                        .value_counts()
                        .sort_index()
                    )

                    x = np.array(
                        cluster[feature]
                        .value_counts()
                        .sort_index()
                        .index
                    )

                    cluster_bar.append([x, y])

            cluster_bar = np.array(cluster_bar)

            # Criação da figura de subplots
            figx, ax = plt.subplots(rows, columns)

            figx.set_size_inches(10.5, 28.5)

            cluster_in_subplot_axis_dict = np.array([
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [1, 1],
                [2, 2]
            ])

            c = 0

            # Plotagem dos gráficos de barras
            for i in range(rows):
                for j in range(columns):

                    if c >= len(cluster_df_list):
                        break

                    ax[i, j].bar(
                        cluster_bar[c, 0],
                        cluster_bar[c, 1],
                        color=col
                    )

                    ax[i, j].tick_params(
                        axis='x',
                        which='major',
                        labelsize=8,
                        rotation=90
                    )

                    ax[i, j].set_title(
                        'Cluster: ' + str(c + 1)
                    )

                    c += 1

        # Vetores para estatísticas
        means = []
        sds = []
        cluster_labels = []

        # Percorre clusters para features contínuas
        for cluster_counter, cluster in enumerate(cluster_df_list):

            if feature in cont_features:

                print(
                    'Resumo estatístico do Cluster '
                    + str(cluster_counter + 1)
                )

                print('\n')

                # Média da feature
                cm = cluster[feature].mean()

                # Desvio padrão
                cs = cluster[feature].std()

                print('Média da feature:', cm)

                print(
                    'Desvio padrão da feature:',
                    cs
                )

                print(
                    'Mediana da feature:',
                    cluster[feature].median()
                )

                print('\n')

                means.append(cm)
                sds.append(cs)

                cluster_labels.append(
                    'C' + str(cluster_counter + 1)
                )

        means = np.array(means)
        sds = np.array(sds)
        cluster_labels = np.array(cluster_labels)

        print('\n')

        print(
            'Distribuição da feature entre os clusters'
        )

        # Visualização para variáveis contínuas
        if feature in cont_features:

            fig, ax7 = plt.subplots()

            ax7.bar(
                cluster_labels,
                means,
                yerr=sds,
                color=sns.color_palette("Set3")
            )

            ax7.tick_params(
                axis='both',
                which='major',
                labelsize=10
            )

            plt.xlabel(feature, fontsize=15)

            plt.show()

        print('\n')
        print('\n')

        # Paleta de cores personalizada
        customPalette_set = sns.set_palette(
            sns.color_palette([
                'lightgray',
                'lightcoral',
                'cornflowerblue',
                'orange',
                'mediumorchid',
                'lightseagreen',
                'olive',
                'chocolate',
                'steelblue',
                'paleturquoise',
                'lightgreen',
                'burlywood',
                'lightsteelblue'
            ])
        )

        # Visualização para variáveis categóricas
        if feature not in cont_features:

            print(
                'Distribuição da feature no embedding UMAP'
            )

            # Aplica dicionário reverso se existir
            if feature in list(rev_dict.keys()):

                umap_data[feature] = np.array(
                    main_data.replace({feature: r})[
                        feature
                    ]
                )

            else:

                umap_data[feature] = np.array(
                    main_data[feature]
                )

            # Plotagem do embedding UMAP
            sns.lmplot(
                x=xName,
                y=yName,
                data=umap_data,
                fit_reg=False,
                legend=True,
                hue=feature,
                scatter_kws={"s": 20},
                palette=customPalette_set
            )

            plt.show()

        print('\n')
        print('\n')