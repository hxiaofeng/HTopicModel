#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from gensim import models
import LDA


# 获取词集合的词向量矩阵，矩阵的每一行对应一个词
def get_words_matrix(words, word2vec_model):
    matrix = []
    dictionary = []
    for word in words:
        try:
            vec = word2vec_model[word]
            matrix.append(vec.tolist())
            dictionary.append(word)
        except KeyError:
            pass
    matrix = np.array(matrix)
    return dictionary, matrix


# 对词集进行聚类
def cluster(matrix, dictionary, n_clusters=2, n_jobs=1):
    cls_model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs)
    cls_model.fit(matrix)
    clusters = dict()
    labels = cls_model.labels_
    for i in range(len(dictionary)):
        clusters.setdefault(labels[i], []).append(dictionary[i])
    return clusters.items()


# 对lda分析生成的词集进行基于词向量的聚类分析
def hierarchical_topic_analyse(lda_model_file, word2vec_model_file, k=1):
    topic2terms = LDA.get_topics_terms(lda_model_file)
    topics = []
    for t in topic2terms:
        topics.append(list(t))
    word2vec_model = models.Word2Vec.load(word2vec_model_file)
    for i in range(k):
        new_topics = []
        for t in topics:
            words = t[-1]
            dictionary, matrix = get_words_matrix(words, word2vec_model)
            clusters = cluster(matrix, dictionary, 2, 10)
            for item in clusters:
                labels = t[:-1]
                labels.extend(list(item))
                new_topics.append(labels)
            del dictionary
            del matrix
            del clusters
        topics = new_topics
    # topic_tree = get_topic_tree(topics)
    # return topic_tree
    return topics


# 对词项集进行聚类分析
def cluster_analyse(words, word2vec_model, k=1):
    dictionary, matrix = get_words_matrix(words, word2vec_model)
    clusters = cluster(matrix, dictionary, 2, 10)
    topics = []
    depth = 0
    for c in clusters:
        topics.append(list(c))
        if len(c) - 1 > depth:
            depth = len(c) - 1
    labels, samples_score = silhouette_samples(topics, word2vec_model)
    clusters_score = silhouette_clusters(labels, samples_score)
    clusters_score.sort(key=lambda x: x[1])
    score = samples_score.mean()
    while clusters_score and depth < k:
        label = clusters_score[0][0]
        del clusters_score[0]
        topic = topics[label]
        subwords = topic[-1]
        dictionary, matrix = get_words_matrix(subwords, word2vec_model)
        clusters = cluster(matrix, dictionary, 2, 10)
        new_topics = []
        for c in clusters:
            new_topic = topic[:-1]
            new_topic.extend(list(c))
            new_topics.append(new_topic)
        new_topics.extend(topics[:label])
        new_topics.extend(topics[label + 1:])
        new_labels, new_samples_score = silhouette_samples(new_topics, word2vec_model)
        new_score = new_samples_score.mean()
        if new_score >= score:
            clusters_score = silhouette_clusters(new_labels, new_samples_score)
            clusters_score.sort(key=lambda x: x[1])
            score = new_score
            topics = new_topics
        for t in topics:
            if len(t) -1 > depth:
                depth = len(t) - 1
    return topics


# 计算聚类结果中每个数据点的轮廓系数
def silhouette_samples(clusters, word2vec_model):
    labels = []
    matrix = []
    for i in range(len(clusters)):
        words = clusters[i][-1]
        _, mat = get_words_matrix(words, word2vec_model)
        for j in range(len(mat)):
            matrix.append(list(mat[j]))
            labels.append(i)
    matrix = np.array(matrix)
    labels = np.array(labels)
    samples_score = metrics.silhouette_samples(matrix, labels)
    return labels, samples_score


# 计算聚类结果中每个簇的轮廓系数
# 用簇内所有数据点的轮廓系数的均值作为这个簇的轮廓系数
def silhouette_clusters(labels, samples_score):
    clusters_score = dict()
    clusters_samples_score = dict()
    for i in range(len(labels)):
        clusters_samples_score.setdefault(labels[i], []).append(samples_score[i])
    for label, scores in clusters_samples_score.items():
        clusters_score[label] = np.array(scores).mean()
    return clusters_score.items()


# 利用带路径的主题列表构建主题树
def get_topic_tree(topics):
    tree = dict()
    for t in topics:
        root = tree
        for label in t[:-2]:
            root.setdefault(label, dict())
            root = root[label]
        assert t[-2] not in root
        root[t[-2]] = t[-1]
    return tree
