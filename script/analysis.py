#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.cluster import KMeans
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
    return clusters


# 对lda分析生成的词集进行基于词向量的聚类分析
def hierarchical_topic_analyse(lda_model_file, word2vec_model_file, k=1):
    topic2terms = LDA.get_topics_terms(lda_model_file)
    if k == 0:
        return topic2terms
    word2vec_model = models.Word2Vec.load(word2vec_model_file)
    topics = []
    for t in topic2terms.items():
        topics.append(list(t))
    for i in range(k):
        new_topics = []
        for t in topics:
            words = t[-1]
            dictionary, matrix = get_words_matrix(words, word2vec_model)
            clusters = cluster(matrix, dictionary, 2, 10)
            for item in clusters.items():
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
