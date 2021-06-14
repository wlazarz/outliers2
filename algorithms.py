from kmodes.kmodes import KModes
import numpy as np

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v,b)*b  for b in basis)
        if (w > 1e-10).any():
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def stirr(X, threshold=0.1):
    X_train=X.copy()
    weights = {}
    weights_sum = []
    keys = []
    for column in X_train:
        uniq_val = X_train[column].value_counts().keys().tolist()
        for v in range(len(uniq_val)):
            df_with_val = X_train.loc[X_train[column] == uniq_val[v]]
            df_with_val.drop([column], axis=1, inplace=True)
            links_count = 0
            for link in df_with_val:
                links_count += len(df_with_val[link].unique())
            weights_sum.append(links_count)
            keys.append((uniq_val[v], column))
            weights[(uniq_val[v], column)] = links_count

    vectors = np.array([weights_sum, weights_sum])

    Q = gram_schmidt(vectors)
    for i in range(len(keys)):
        weights[keys[i]] = Q[len(Q) - 1][i]

    dict_values = [[0] * len(weights_sum), list(weights.values()), list(weights.values())]
    iter = 0
    last_vector_idx = 2
    while list(map(abs, np.round(dict_values[last_vector_idx - 1], decimals=4))) != list(
            map(abs, np.round(dict_values[last_vector_idx], decimals=4))) \
            or list(map(abs, np.round(dict_values[last_vector_idx - 1], decimals=4))) != list(
        map(abs, np.round(dict_values[last_vector_idx - 2], decimals=4))):
        weights_sum = []
        keys = []
        for k, l in weights:
            df_with_val = X_train.loc[X_train[l] == k]
            for val, col in weights:
                df_with_val.loc[df_with_val[col] == val, col] = weights[val, col]
            df_temp = df_with_val.drop([l], axis=1, inplace=False)
            df_temp['max_value'] = df_temp.max(axis=1)
            weights_sum.append(df_temp['max_value'].sum())
            keys.append((k, l))

        vectors = np.vstack([vectors, weights_sum])
        Q = gram_schmidt(vectors)
        for i in range(len(keys)):
            weights[keys[i]] = round(Q[len(Q) - 1][i], 10)

        dict_values.append(list(weights.values()))
        last_vector_idx = len(vectors) - 1

        iter += 1

    X_weights = X_train
    for val, col in weights:
        X_weights.loc[X_weights[col] == val, col] = weights[val, col]
    wsum = X_weights.sum(axis=1)
    X_weights['sum_of_weights'] = wsum
    cut_val = sorted(list(map(abs, X_weights['sum_of_weights'])))[round(threshold * len(X_weights))]
    labels = []
    for index, value in X_weights['sum_of_weights'].items():
        if abs(value) < cut_val:
            labels.append(-1)
        else:
            labels.append(1)
    return wsum, labels, weights


def stirr_calculate_output(X, wsum, threshold):
    
    X_weights = X.copy()
    
    try:
        X_weights['sum_of_weights'] = wsum
        cut_val = sorted(list(map(abs, X_weights['sum_of_weights'])))[round(threshold * len(X_weights))]
        labels = []
        for index, value in X_weights['sum_of_weights'].items():
            if abs(value) < cut_val:
                labels.append(-1)
            else:
                labels.append(1)
        return labels
            
    except:
        return "weights not calculated"
        
        
    return wsum, labels, weights
    
    

def jaccard_measure(a, b):
    intersect = list(set(a) & set(b))
    union = list(set(a).union(set(b)))

    if len(intersect) == 0:
        return 0.0
    else:
        return len(intersect) * 1.0 / len(union)

def rock(X, k, theta):
    
    data=X.copy().to_numpy()
    func = 1.0 + 2.0 * ((1.0 - theta) / (1.0 + theta))
    len_X = len(data)
    links_matrix = [[0]*len_X for i in range(len_X)]
    for i in range(0, len_X):
        for j in range(i + 1, len_X):
            distance = jaccard_measure(data[i], data[j])
            if (distance >= theta):
                links_matrix[i][j] = 1
                links_matrix[j][i] = 1
    
    clusters = [[index] for index in range(len(data))]
    while len(clusters) > k:
        goodness_measure_max = 0.0
        indexes = [-1, -1]

        for i in range(0, len(clusters)):
            for j in range(i + 1, len(clusters)):
                number_links = 0
                for index1 in clusters[i]:
                    for index2 in clusters[j]:
                        number_links += links_matrix[index1][index2]
                devider = (len(clusters[i]) + len(clusters[j])) ** func - len(clusters[i]) ** func - len(
                    clusters[j]) ** func
                goodness_measure = number_links / devider
                if (goodness_measure > goodness_measure_max):
                    goodness_measure_max = goodness_measure
                    indexes = [i, j]
        if (indexes != [-1, -1]):
            clusters[indexes[0]] += clusters[indexes[1]]
            clusters.pop(indexes[1])
        else:
            break
        
    labels = [0]*len(X)
    clusters_out = [0]*len(X)
    cluster_num = 1
    for c in clusters:
        if len(c) < 2:
            for i in c:
                labels[i] = -1
                clusters_out[i] = 0
        else:
            for i in c:
                labels[i] = 1
                clusters_out[i] = cluster_num
        cluster_num += 1
            
    return clusters, clusters_out, labels

import math
def kmodes(X, k, outlier_fraction=0.05):
    X_dataset=X.copy()
    cat_col = list(X_dataset.columns)
    model = KModes(n_clusters=k, n_init=50, init="random")
    clusters = model.fit_predict(X_dataset, categorical = cat_col)
    centroids = model.cluster_centroids_
    labels = model.labels_
    data=X_dataset.to_numpy()
    
    distances = []
    for i in range(len(clusters)):
        distances.append(calculate_distance(centroids[clusters[i]], data[i]))
    
    
    list1 = sorted(distances, reverse = True)
    threshold = list1[math.ceil(len(list1)*outlier_fraction)]
    outliers = []
    for i in distances:
        if i < threshold:
            outliers.append(1)
        else:
            outliers.append(-1)
    return clusters, outliers, centroids

def kmodes_calculate_outliers(X, clusters, centroids, outlier_fraction):
    data = X.copy().to_numpy()
    distances = []
    for i in range(len(clusters)):
        distances.append(calculate_distance(centroids[clusters[i]], data[i]))
    
    
    list1 = sorted(distances, reverse = True)
    threshold = list1[math.ceil(len(list1)*outlier_fraction)]
    outliers = []
    for i in distances:
        if i <= threshold:
            outliers.append(1)
        else:
            outliers.append(-1)
    return outliers
     
def calculate_distance(x, y):
    dist=0
    for i in range(len(x)):
        if x[i]!=y[i]: dist+=1
            
    return dist